import copy, os, logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .config import Config
from .utils import set_seed, save_outputs, msg, count_params
from .preprocess import get_dataloaders


class EMA:
    """
    Exponential Moving Average
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_avg.clone()

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        if not self.backup:
            return
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class FFN(nn.Module):
    """Feedforward Network with Residual Connection."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout=Config.dropout,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        return x + self.model(self.ln(x))  # Pre-norm


class ResidualBlock(nn.Module):
    """Multiple Residually-Connected Sequential MLP Blocks."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_residual_layers=Config.num_residual_layers,
        dropout=Config.dropout,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[
                FFN(hidden_dim, hidden_dim * 2, dropout)
                for _ in range(num_residual_layers)
            ]  # won't share parameters
        )

        self.ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.blocks(x)
        x = self.ln(x)
        x = self.out_proj(x)
        return x


class PoolNetwork(nn.Module):
    """Strategy for pooling embeddings along temporal axis."""

    def __init__(
        self,
        hidden_dim,
        nhead=Config.time_nheads,
        dropout=Config.dropout,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        B, F, f = x.shape
        if Config.pooling_method == "last":
            return x[:, -1, :]
        else:
            raise NotImplementedError("Pooling method not supported as of 12/2")


class PFTransformer(nn.Module):
    """Play-level Spatio-Temporal Transformer."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        time_nheads=Config.time_nheads,
        player_nheads=Config.player_nheads,
        dim_feedforward=Config.dim_feedforward,
        num_time_encoder_layers=Config.num_time_encoder_layers,
        num_player_encoder_layers=Config.num_player_encoder_layers,
        dropout=Config.dropout,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # --- Encoder 1: over time (frames) ---
        self.pos_embed = nn.Parameter(
            torch.randn(size=(Config.window_size, hidden_dim))
        )
        self.drop = nn.Dropout(dropout)
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=time_nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.time_encoder = nn.TransformerEncoder(
            time_encoder_layer,
            num_layers=num_time_encoder_layers,
        )

        # --- Encoder 2: over players ---
        player_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=player_nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.player_encoder = nn.TransformerEncoder(
            player_encoder_layer,
            num_layers=num_player_encoder_layers,
        )

        self.mlp = ResidualBlock(
            input_dim=hidden_dim,
            hidden_dim=Config.mlp_hidden_dim,
            output_dim=Config.horizon * 2,
        )

        self.pool = PoolNetwork(hidden_dim=hidden_dim)

    def forward(self, x, key_padding_mask):
        """
        Args:
            x: Player-frame features of shape (B, P, T, F)
            key_padding_mask: Boolean tensor with shape (B, P), where True
                indicates a real player and False indicates 0 padding.
        """
        B, P, T, F = x.shape
        B, P = key_padding_mask.shape

        x_proj = self.input_projection(x)
        x_proj = self.drop(x_proj + self.pos_embed).reshape(B * P, T, self.hidden_dim)

        x_t = self.time_encoder(x_proj)  # time encoder

        x_t = self.pool(x_t)  # (B * P, f)
        num_features = x_t.shape[-1]
        x_t = x_t.reshape(B, P, num_features)

        x_f = self.player_encoder(
            x_t, src_key_padding_mask=(1 - key_padding_mask.int()).bool()
        )

        y_pred = self.mlp(x_f)
        y_pred = y_pred.reshape(B, P, Config.horizon, 2)
        y_pred = y_pred.cumsum(dim=2)  # cumsum

        return y_pred


class TemporalHuber(nn.Module):
    """Implements Temporal Huber Loss."""

    def __init__(self, delta=0.5, time_decay=Config.time_decay):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay

    def forward(self, y_pred, y_true, y_mask=None):
        B, P, T, _ = y_pred.shape
        a = torch.abs(y_pred - y_true)
        L_delta = torch.where(
            a <= self.delta, 0.5 * (a**2), self.delta * (a - 0.5 * self.delta)
        )

        if self.time_decay > 0:
            T = y_pred.size(2)
            t = torch.arange(T, device=Config.device).float()
            weight = torch.exp(-self.time_decay * t).view(1, 1, T, 1)
            if y_mask is not None:
                # Temporal weighting applied here to ensure correct normalization
                y_mask = y_mask * weight.squeeze(-1)

        return L_delta, y_mask


class LossFunction(nn.Module):
    """Loss Function Router."""

    def __init__(self):
        super().__init__()
        if Config.loss_type == "huber":
            self.loss = TemporalHuber()
        else:
            raise NotImplementedError(f"{Config.loss_type} hasn't been implemented.")

    def forward(self, x, y, m=None):
        """
        Args:
            x: Model predictions with shape (B, N, 2)
            y: Ground truth values with shape (B, N, 2)
            m: Boolean mask of shape (B, N)
        """
        loss, mask = self.loss(x, y, m)

        if mask is not None:
            assert mask.sum() > 0
            loss = (loss * mask[:, :, :, None]).sum() / (mask.sum() * 2)

        return loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float | None = None,
    scheduler=None,
    ema=None,
) -> float:
    """Train over one epoch and return mean MSE over the dataset."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X, K, y, m in tqdm(loader, total=len(loader), desc="Training: ", leave=False):
        X = X.to(device)
        K = K.to(device).bool()
        y = y.to(device)
        m = m.to(device).bool()

        optimizer.zero_grad(set_to_none=True)
        preds = model(X, K)
        loss = criterion(preds, y, m)
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()
        if Config.use_scheduler:
            scheduler.step()
        if Config.use_ema:
            ema.update(model)

        B = X.size(0)
        total_loss += loss.item() * B
        total_samples += B

    # Take the average loss over all samples
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model: returns (MSE, RMSE) on the dataset."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    total_sq_err = 0.0
    total_unmask = 0.0

    for X, K, y, m in tqdm(loader, total=len(loader), desc="Training: ", leave=False):
        X = X.to(device, non_blocking=True)
        K = K.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)

        preds = model(X, K)
        loss = criterion(preds, y, m)

        total_sq_err += ((preds - y) ** 2 * m[:, :, :, None]).sum()
        total_unmask += m.sum() * 2

        B = X.size(0)
        total_loss += loss.item() * B
        total_samples += B

    mse = total_loss / max(total_samples, 1)
    rmse = torch.sqrt(total_sq_err / total_unmask).item()

    return mse, rmse


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    seed: int,
    verbose: bool = True,
    fold_id=None,
) -> nn.Module:
    """Train PFTransformer on the train/val split specified by train_loader/val_loader (corresponding to fold_id)."""

    msg(
        f"Training has begun. Model of {type(model)} with {count_params(model)} parameters"
    )
    print(model)

    # Reset logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(Config.SAVE_DIR, "training.log"),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    device = torch.device(Config.device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.lr,
        weight_decay=Config.weight_decay,
    )

    ema = None
    if Config.use_ema:
        ema = EMA(model, decay=0.996)  # average over last 250 steps

    scheduler = None
    if Config.use_scheduler:
        num_training_steps = Config.num_epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=Config.num_warmup_steps,
        )

    criterion = LossFunction()

    best_val_rmse = float("inf")
    best_state = None

    train_loss_list = []
    val_loss_list = []
    val_rmse_list = []

    for epoch in range(Config.num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            clip_grad_norm=Config.clip_grad_norm,
            scheduler=scheduler,
            ema=ema,
        )

        if Config.use_ema:
            ema.apply_shadow(model)

        val_loss, val_rmse = evaluate(model, val_loader, criterion, device)

        if verbose:
            print(
                f"Epoch {epoch + 1}/{Config.num_epochs} "
                f"- train loss: {train_loss} "
                f"- val loss: {val_loss} "
                f"- val RMSE: {val_rmse} "
            )
            logger.info(
                f"Epoch {epoch + 1}/{Config.num_epochs} "
                f"- train loss: {train_loss} "
                f"- val loss: {val_loss} "
                f"- val RMSE: {val_rmse} "
            )
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            val_rmse_list.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())  # do NOT return references

        if Config.use_ema:
            ema.restore(model)

    # Save loss curves to a plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.plot(train_loss_list, c="r")
    plt.plot(val_loss_list, c="b")
    plt.legend(["Training Loss", "Validation Loss"])
    plt.savefig(Config.SAVE_DIR / f"loss_fold{fold_id}.png")

    plt.figure(figsize=(6, 6))
    plt.plot(val_rmse_list)
    plt.savefig(Config.SAVE_DIR / f"rmse_fold{fold_id}.png")

    if best_state is not None:
        # torch.save(best_state, "best_mlp.pth")
        model.load_state_dict(best_state)
        best_mlp = save_outputs(seed, model, Config.SAVE_DIR, fold_id=fold_id)
        if verbose:
            print(f"Best val RMSE: {best_val_rmse:.4f} (weights saved to '{best_mlp}')")
            logger.info(
                f"Best val RMSE: {best_val_rmse:.4f} (weights saved to '{best_mlp}')"
            )

    return model, best_val_rmse


def train_all_folds(X, K, y, m, folds, seed, verbose=True):
    """Train PFTransformer on all Cross-Validation folds."""

    models = []
    fold_sm = 0.00

    print(f"Training {len(folds)} folds ...")
    for i in range(len(folds)):
        print(f"Training the {i}-th fold ...")
        train_indices = folds[i]["train_idx"]
        val_indices = folds[i]["val_idx"]
        X_train = X[train_indices, :, :, :]  # (B, P, F, f)
        K_train = K[train_indices, :]  # (B, P)
        y_train = y[train_indices, :, :, :]  # (B, P, Config.horizon, 2)
        m_train = m[train_indices, :, :]  # (B, P, Config.horizon)
        X_val = X[val_indices, :, :, :]
        K_val = K[val_indices, :]
        y_val = y[val_indices, :, :, :]
        m_val = m[val_indices, :, :]

        train_loader, val_loader = get_dataloaders(
            X_train,
            K_train,
            y_train,
            m_train,
            X_val,
            K_val,
            y_val,
            m_val,
        )

        # Re-initialize model
        model = PFTransformer(
            input_dim=Config.input_dim,
            hidden_dim=Config.st_hidden_dim,
            output_dim=Config.output_dim,
            time_nheads=Config.time_nheads,
            player_nheads=Config.player_nheads,
            dim_feedforward=Config.dim_feedforward,
            num_time_encoder_layers=Config.num_time_encoder_layers,
            num_player_encoder_layers=Config.num_player_encoder_layers,
            dropout=Config.dropout,
        )

        result, fold_rmse = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            seed=seed,
            verbose=verbose,
            fold_id=i,
        )

        models.append(result)
        fold_sm += fold_rmse
        print(f"Finished training {i}-th fold")

        if not Config.all_folds:  # only trained one fold
            print(f"Early stopping.")
            break

    print(f"\n[INFO] All folds finished, average RMSE over folds: {fold_sm / (i + 1)}")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(Config.SAVE_DIR, "training.log"),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"All folds finished, average RMSE over folds: {fold_sm / (i + 1)}")

    return models

import argparse, sys, os
import torch
import numpy as np
import joblib
from pathlib import Path
from src.utils import set_folder, set_seed, get_persistent_directory
from src.config import Config
from src.model import PFTransformer, train_model, train_all_folds
from src.preprocess import get_dataloaders


def main(args):

    sys.path.append(os.getcwd())
    # torch.autograd.set_detect_anomaly(True)

    RANDOM_SEED = 888
    set_seed(RANDOM_SEED)

    Config.BASE_DIR = Path(os.getcwd())
    Config.G_DATA_DIR = Path(args.train_data_dir)
    Config.TRAIN_PATH = Path(Config.DATA_DIR) / "train"
    Config.OUTPUT_DIR = Path(args.output_dir)
    Config.DEBUG = False
    Config.TRAIN = True
    set_folder()

    train_data_path = Config.G_DATA_DIR / f"train_data_{Config.G_DATA_VERSION}.npz"
    val_data_path = Config.G_DATA_DIR / f"val_data_{Config.G_DATA_VERSION}.npz"
    scaler_path = Config.G_DATA_DIR / f"scaler_{Config.G_DATA_VERSION}.pkl"
    fold_path = Config.G_DATA_DIR / f"gkf_split_{Config.G_DATA_VERSION}.pkl"

    SEEDS = [888, 3407, 42, 0, 1]

    if (
        Config.split_type == "tts"
    ):  # Training on a specific train/val split (currently unused)
        assert os.path.exists(train_data_path) and os.path.exists(val_data_path), (
            f"Run preprocessing loop first."
        )
        Xt = np.load(train_data_path)
        Xv = np.load(val_data_path)
        import joblib
        # scaler = joblib.load(scaler_path)

        X_train, K_train, y_train, m_train = (
            Xt["X_train"],
            Xt["K_train"],
            Xt["y_train"],
            Xt["m_train"],
        )
        X_val, K_val, y_val, m_val = Xv["X_val"], Xv["K_val"], Xv["y_val"], Xv["m_val"]

        from src.preprocess import get_dataloaders

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

        input_dim = len(Config.feature_cols)

        print(f"#samples train: {len(train_loader.dataset)}")
        print(f"#samples val:   {len(val_loader.dataset)}")
        print(f"#features:      {input_dim}")
        print(f"#train batches: {len(train_loader)}")
        print(f"#eval batches:  {len(val_loader)}")

        for seed in SEEDS:
            print(f"\n\nTraining on seed {seed} ...")
            set_seed(seed)
            print(f"Random seed is set!")
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
            print(f"Training Model: ")
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                seed=seed,
                verbose=True,
                fold_id=0,  # assume it's only "1 fold"
            )

    else:  # Training on all folds
        train_loaded = np.load(train_data_path)
        X_train = train_loaded["X_train"]
        K_train = train_loaded["K_train"]
        y_train = train_loaded["y_train"]
        m_train = train_loaded["m_train"]
        import joblib

        fold_indices = joblib.load(fold_path)

        for seed in SEEDS:
            print(f"\n\nTraining on seed {seed} ...")
            set_seed(seed)
            print(f"Random seed is set!")
            train_all_folds(
                X_train,
                K_train,
                y_train,
                m_train,
                fold_indices,
                seed=seed,
                verbose=True,
            )


if __name__ == "__main__":
    # CLI Support
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-dir",
        type=str,
        help="Path to folder containing .npz data",
        default=get_persistent_directory() / ".nfl-silver-solution" / "generated_data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to which program will save model weights",
        default=get_persistent_directory()
        / ".nfl-silver-solution"
        / "training_artifacts",
    )
    args = parser.parse_args()
    main(args)

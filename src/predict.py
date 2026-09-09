import os
import torch
import numpy as np
import pandas as pd

from .config import Config
from .model import PFTransformer
from .preprocess import load_and_preprocess, preprocess

_loaded_models = None


def load(test_input, test, time_tag, seeds, scaler_path, device):
    """Loads the necessary model ensemble and preprocesses the testing data."""

    # Data
    test_, test_output = preprocess(test_input, test)
    X_test, K_test, scaler = load_and_preprocess(test_, test_output, scaler_path)

    global _loaded_models
    if _loaded_models is None:
        _loaded_models = []
        for seed in seeds:
            for i in range(0, 5):
                save_dir = (
                    Config.OUTPUT_DIR / time_tag / f"seed_{seed}" / f"model_fold{i}.pt"
                )
                if not os.path.exists(save_dir):
                    break
                state_dict = torch.load(
                    save_dir, weights_only=False, map_location=Config.device
                )
                model = PFTransformer(
                    input_dim=Config.input_dim,
                    output_dim=Config.output_dim,
                    hidden_dim=Config.st_hidden_dim,
                    time_nheads=Config.time_nheads,
                    player_nheads=Config.player_nheads,
                    dim_feedforward=Config.dim_feedforward,
                    num_time_encoder_layers=Config.num_time_encoder_layers,
                    num_player_encoder_layers=Config.num_player_encoder_layers,
                    dropout=Config.dropout,
                ).to(device)
                model.load_state_dict(state_dict)

                # Switch to evaluation mode
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(False)

                _loaded_models.append(model)

    return X_test, K_test, scaler, test_, test_output


def ensemble(input1, input2):
    """Averages outputs over model ensemble given input arguments (input1, input2)."""
    model_outputs = []
    for m in _loaded_models:
        output = m(input1, input2)
        model_outputs.append(output)
    return torch.stack(model_outputs, dim=0).mean(dim=0)


def post_processing(y_pred: torch.tensor, dir_raw):
    """Transform predictions back into their original (unrotated) directions."""

    # normalize dir_raw
    direction = str(dir_raw).strip().lower()

    if Config.FLIP_LEFT_RIGHT and direction == "right":
        y_pred[:, :, 0] = Config.FIELD_X_MAX - y_pred[:, :, 0]
        if Config.FLIP_Y:
            y_pred[:, :, 1] = Config.FIELD_Y_MAX - y_pred[:, :, 1]

    # No need to clamp outputs:
    # y_pred[:, 0] = y_pred[:, 0].clamp(0, Config.FIELD_X_MAX)
    # y_pred[:, 1] = y_pred[:, 1].clamp(0, Config.FIELD_Y_MAX)
    return y_pred


def pf_inference(test, test_input, time_tag, seeds, device=Config.device):
    """Inference Loop."""

    # Loading Data
    scaler_path = Config.G_DATA_DIR / f"scaler_{Config.G_DATA_VERSION}.pkl"
    X_test, K_test, scaler, test_, test_output = load(
        test_input, test, time_tag, seeds, scaler_path, device
    )  # test_ and test_output: both unsorted (in original order)
    y_pred = []
    test_input = test_  # renaming

    # Getting indices and play-level info for each (game_id, play_id) pair
    ids = test_output.sort_values(["game_id", "play_id"])[
        ["game_id", "play_id"]
    ].drop_duplicates()  # sorted index
    ids = [tuple(x) for x in ids.to_numpy()]
    num_frames = test_output.groupby(["game_id", "play_id", "nfl_id"])["frame_id"].max()
    play_dir = (
        test_output.groupby(["game_id", "play_id"])["play_direction"].first().tolist()
    )

    # Sorting the input
    test_input = test_input.sort_values(
        ["game_id", "play_id", "player_to_predict", "nfl_id", "frame_id"],
        ascending=[True, True, False, True, True],
    )  # sorted order

    # Extracting the last frame
    last_idx = test_input.groupby(["game_id", "play_id", "nfl_id"], sort=False)[
        "frame_id"
    ].idxmax()
    last_frame_data = test_input.loc[last_idx].set_index(["game_id", "play_id"])

    if Config.SANITY_CHECKS:
        # Sanity Check: after sorting input_data by player_to_predict DESC
        for (gid, pid), df_play in test_input.groupby(
            ["game_id", "play_id"], sort=False
        ):
            player_order = df_play.drop_duplicates("nfl_id")
            flags = player_order["player_to_predict"].to_numpy()
            if not np.all(flags == np.sort(flags)[::-1]):
                raise AssertionError(
                    f"player_to_predict ordering is incorrect for play {(gid, pid)}: got {flags}"
                )

        # Sanity Check 2: check if all of the `ids` exist in the index of the last_frame_data
        indices = last_frame_data.index.tolist()
        for idx in ids:
            try:
                assert idx in indices
                thingy = last_frame_data.loc[idx]
            except Exception as e:
                print(f"specifically {idx}")
                raise ValueError("`indices` does not include all of `ids`")

    # Loop through the X_test order (completely sorted by game_id, play_id)
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            gid, pid = ids[i]

            input1 = (
                torch.tensor(X_test[i]).unsqueeze(0).to(device).to(torch.float32)
            )  # (1, P, F, f)
            input2 = torch.tensor(K_test[i]).unsqueeze(0).to(device).bool()
            output = ensemble(input1, input2)

            offset = torch.tensor(
                last_frame_data.loc[(gid, pid)][["x", "y"]].to_numpy(),  # (P, 2)
            ).to(device)  # (P, 2)
            offset = offset.unsqueeze(1)  # (P, 1, 2)

            test_play = test_output[
                (test_output.game_id == gid) & (test_output.play_id == pid)
            ]
            if (
                test_play.shape[0] == 0
            ):  # if (gid, pid) doesn't exist in the test_output, then skip
                continue

            test_play = test_play.sort_values(
                ["nfl_id", "frame_id"]
            )  # no need to sort player_to_predict, since it's always True
            P_pred = test_play["nfl_id"].nunique()  # number of players to predict

            output = output[
                :, :P_pred, :, :
            ]  # capping at the first P_pred players (from our sorted assumption)
            offset = offset[:P_pred]
            output = (output + offset).squeeze(0)  # (P_pred, horizon, 2)
            output = post_processing(output, play_dir[i])

            for j, nid in enumerate(
                test_play["nfl_id"].drop_duplicates().values
            ):  # for every nfl_id (in the sorted order)
                F = int(num_frames.loc[(gid, pid, nid)])
                y = output[j, :, :]  # (horizon, 2)
                # take the i-th player --> uses the sorted assumption

                if F > y.shape[0]:  # (1, 2)
                    last_predicted = (
                        y[-1, :].unsqueeze(0).repeat(F - y.shape[0], 1).to(device)
                    )  # (F - y.shape[1], 1)
                    y = torch.cat([y, last_predicted], dim=0)  # (F, 2)
                else:
                    y = y[:F, :]

                y_pred.append(
                    pd.DataFrame(
                        {
                            "game_id": [gid] * F,
                            "play_id": [pid] * F,
                            "nfl_id": [nid] * F,
                            "frame_id": test_play[test_play.nfl_id == nid][
                                "frame_id"
                            ].values,
                            "x": y[:, 0].detach().cpu().numpy(),
                            "y": y[:, 1].detach().cpu().numpy(),
                        }
                    )
                )

    predicts = pd.concat(y_pred, axis=0)
    prediction = test_output.merge(
        predicts, on=["game_id", "play_id", "nfl_id", "frame_id"], how="left"
    )[["x", "y"]].reset_index(drop=True)  # how='left' guarantees index preservation
    return prediction

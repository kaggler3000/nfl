import os, random
import subprocess
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from .config import Config


def get_persistent_directory():
    return Path("/workspace") if Path("/workspace").is_dir() else Path.home()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_folder():
    from datetime import datetime

    utc_now = datetime.now()
    Config.TIME_TAG = utc_now.strftime("%Y_%m_%d_%H_%M_%S")
    Config.SAVE_DIR = Config.OUTPUT_DIR / Config.TIME_TAG
    print(f"Config.TIME_TAG: {Config.TIME_TAG}")
    print(f"Config.SAVE_DIR: {Config.SAVE_DIR}")
    if not os.path.exists(Config.SAVE_DIR):
        os.makedirs(Config.SAVE_DIR, exist_ok=True)


def msg(message, padding=1):
    print("-" * (len(message) + 2 * (padding + 1)))
    print("|" + " " * padding + message + " " * padding + "|")
    print("-" * (len(message) + 2 * (padding + 1)))


def count_params(model):
    cnt = 0
    for param in model.parameters():
        if param.requires_grad:
            cnt += param.numel()
    return cnt


def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["game_id", "play_id", "nfl_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["game_id", "play_id", "nfl_id"])
    for col in ["game_id", "play_id", "nfl_id"]:
        df[col] = df[col].astype("int64")
    return df


def flip_angle(angle):
    if Config.FLIP_Y:
        return (angle + 180.0) % 360.0
    else:
        return (180.0 - angle) % 360.0


def standardize_play_direction(
    input_df: pd.DataFrame, output_df: pd.DataFrame
) -> pd.DataFrame:
    input_df = input_df.copy()
    output_df = output_df.copy()

    # Fixing input
    input_df["play_direction"] = input_df["play_direction"].apply(
        lambda x: str(x).strip().lower()
    )
    mask = input_df["play_direction"] == "right"

    input_df.loc[mask, "x"] = Config.FIELD_X_MAX - input_df.loc[mask, "x"]
    input_df.loc[mask, "ball_land_x"] = (
        Config.FIELD_X_MAX - input_df.loc[mask, "ball_land_x"]
    )
    if Config.FLIP_Y:
        input_df.loc[mask, "y"] = Config.FIELD_Y_MAX - input_df.loc[mask, "y"]
        input_df.loc[mask, "ball_land_y"] = (
            Config.FIELD_Y_MAX - input_df.loc[mask, "ball_land_y"]
        )

    for col in ("dir", "o"):
        if col in input_df.columns:
            input_df.loc[mask, col] = input_df.loc[mask, col].apply(flip_angle)

    # Mapping play_direction
    dir_map = input_df[
        ["game_id", "play_id", "nfl_id", "play_direction"]
    ].drop_duplicates()
    output_df = output_df.merge(
        dir_map, on=["game_id", "play_id", "nfl_id"], how="left"
    )

    # Fixing output
    right_mask_out = output_df["play_direction"] == "right"

    if "x" in output_df.columns and "y" in output_df.columns:
        output_df.loc[right_mask_out, "x"] = (
            Config.FIELD_X_MAX - output_df.loc[right_mask_out, "x"]
        )
        if Config.FLIP_Y:
            output_df.loc[right_mask_out, "y"] = (
                Config.FIELD_Y_MAX - output_df.loc[right_mask_out, "y"]
            )

    return input_df, output_df


def prepare_targets(y: list[np.array], max_h=Config.horizon):
    """
    Input:
     - y: the list of ndarrays that represent the (dx, dy) targets of the model

    Output:
     - y_: y but padded with 0s
     - masks: masks to prevent loss from calculating extraneous time positions
    """
    # y[i].shape = (P, L_i, 2)
    tot = []
    msks = []
    for arr in y:
        P = arr.shape[0]
        mask = np.zeros((P, max_h))
        mask[:, : arr.shape[-2]] = 1.0  # float mask for loss function
        zr = np.zeros(shape=(P, max_h - arr.shape[1], 2))
        arr = np.concatenate([arr, zr], axis=1)  # (max_h, 2)

        # P, horizon, 2 = arr.shape
        # P, horizon = mask.shape
        if arr.shape[0] < Config.max_players:
            arr = np.concatenate(
                [
                    arr,
                    np.zeros(
                        shape=(Config.max_players - P, arr.shape[1], arr.shape[2])
                    ),
                ],
                axis=0,
            )
            mask = np.concatenate(
                [mask, np.zeros(shape=(Config.max_players - P, mask.shape[1]))], axis=0
            )

        tot.append(arr)
        msks.append(mask)

    y_, masks = np.stack(tot, axis=0), np.stack(msks, axis=0)
    return y_, masks


def load_input_output():

    # getting input file paths
    train_input_files = [
        Config.DATA_DIR / f"train/input_2023_w{w:02d}.csv"
        for w in range(1, 19 if not Config.DEBUG else 2)
    ]
    train_output_files = [
        Config.DATA_DIR / f"train/output_2023_w{w:02d}.csv"
        for w in range(1, 19 if not Config.DEBUG else 2)
    ]

    # concatenate training data w01 to w18 together
    train_input = pd.concat(
        [pd.read_csv(f) for f in train_input_files if os.path.exists(f)],
        ignore_index=True,
    )
    train_output = pd.concat(
        [pd.read_csv(f) for f in train_output_files if os.path.exists(f)],
        ignore_index=True,
    )

    # https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/discussion/611647#3310487
    bad_game_id = 2023091100
    bad_play_id = 3167

    before_in = len(train_input)
    before_out = len(train_output)

    train_input = train_input[
        ~(
            (train_input["game_id"] == bad_game_id)
            & (train_input["play_id"] == bad_play_id)
        )
    ]
    train_output = train_output[
        ~(
            (train_output["game_id"] == bad_game_id)
            & (train_output["play_id"] == bad_play_id)
        )
    ]

    print("Filtered input rows: ", before_in - len(train_input))
    print("Filtered output rows: ", before_out - len(train_output))

    return train_input, train_output


def save_outputs(seed, model, base_dir, fold_id=None):
    seed_dir = base_dir / f"seed_{seed}"
    os.makedirs(seed_dir, exist_ok=True)
    # joblib.dump(scaler, sdir / f"scaler.pkl")
    if fold_id is not None:
        torch.save(model.state_dict(), seed_dir / f"model_fold{fold_id}.pt")
    else:
        torch.save(model.state_dict(), seed_dir / f"model.pt")

    subprocess.run(
        [
            "cp",
            "-r",
            Config.BASE_DIR / "src",
            Config.SAVE_DIR,
        ]
    )

    return str(seed_dir / f"model.pt")

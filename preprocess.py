import argparse, sys, os
import torch
import numpy as np
import joblib
from pathlib import Path
from src.utils import set_folder, set_seed, get_persistent_directory
from src.config import Config
from src.model import PFTransformer, train_model, train_all_folds
from src.preprocess import preprocess, load_and_preprocess


def main(args):
    Config.BASE_DIR = Path(os.getcwd())
    Config.DATA_DIR = Path(args.data_dir)
    Config.TRAIN_PATH = Config.DATA_DIR / "train"
    Config.TRAIN = True  # this must be set true during data processing
    Config.G_DATA_DIR = Path(args.train_data_dir)
    train_data_path = Config.G_DATA_DIR / f"train_data_{Config.G_DATA_VERSION}.npz"
    val_data_path = Config.G_DATA_DIR / f"val_data_{Config.G_DATA_VERSION}.npz"
    scaler_path = Config.G_DATA_DIR / f"scaler_{Config.G_DATA_VERSION}.pkl"
    fold_path = Config.G_DATA_DIR / f"gkf_split_{Config.G_DATA_VERSION}.pkl"
    Config.G_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(
        train_data_path
    ):  # comment out if preprocessing needs to be completed again (updating new features)
        if Config.split_type == "gkf":
            _input_df, _output_df = preprocess()
            X, K, y, m, scaler, fold_indices = load_and_preprocess(
                _input_df, _output_df
            )
            joblib.dump(scaler, scaler_path)
            joblib.dump(fold_indices, fold_path)
            np.savez(train_data_path, X_train=X, K_train=K, y_train=y, m_train=m)
        else:  # tts, train_test_split
            X_train, K_train, y_train, m_train, X_val, K_val, y_val, m_val, scaler = (
                load_and_preprocess()
            )
            np.savez(
                train_data_path,
                X_train=X_train,
                K_train=K_train,
                y_train=y_train,
                m_train=m_train,
            )
            np.savez(val_data_path, X_val=X_val, K_val=K_val, y_val=y_val, m_val=m_val)
            joblib.dump(scaler, scaler_path)


if __name__ == "__main__":
    # CLI Support
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
        help="Path to Official 2026 NFL Big Data Bowl dataset.",
    )
    parser.add_argument(
        "--train-data-dir",
        type=str,
        help="Folder to which preprocessed data should be saved.",
        default=get_persistent_directory() / ".nfl-silver-solution" / "generated_data",
    )
    args = parser.parse_args()
    main(args)

import os, argparse, sys, shutil, json, kagglehub
from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))  # repository root
from src.utils import get_persistent_directory
from src.config import Config
from functools import wraps
from requests.exceptions import HTTPError


def permission_loop(msg):
    while 1:
        permission = str(input(msg.strip() + " ")).strip().lower()
        if permission in ["", "n", "no"]:
            return False
        elif permission in ["y", "yes"]:
            return True


# Main loop
def main(args):
    assert args.timestamp is not None, f"Must provide a submission timestamp"
    scaler_path = Path(args.train_data_dir) / f"scaler_{Config.G_DATA_VERSION}.pkl"
    checkpoint_directory = Path(args.output_dir) / args.timestamp
    src_directory = checkpoint_directory / "src"

    def upload_dataset(
        path: Path,
        destination: Path,
        name: str,
        version_notes: str = f"Updated Dataset for {args.timestamp}",
    ):

        closure = destination / name
        permission_loop(f"Create directory at: '{closure}'? [y/N]") and closure.mkdir(
            parents=True, exist_ok=True
        )
        new_path = closure / path.parts[-1]
        permission_loop(f"Copy '{path}' to '{new_path}'? [y/N]") and (
            (path.is_file() and shutil.copy2(path, new_path))
            or shutil.copytree(path, new_path, dirs_exist_ok=True)
        )
        outer_directory = closure
        permission_loop(
            f"Create Kaggle dataset at {f'{args.username}/{name}'} from '{str(outer_directory)}'? [y/N]"
        ) and kagglehub.dataset_upload(
            f"{args.username}/{name}",
            str(outer_directory),
            version_notes=version_notes,
        )

    for path, dest, dataset_ref in zip(
        [scaler_path, checkpoint_directory, src_directory],
        [args.train_data_dir, args.output_dir, args.output_dir],
        # ['Scaler.pkl for NFL Big Data Bowl', 'Model weights for NFL Big Data Bowl', 'Source code for NFL Big Data Bowl'],
        ["nfl-scaler-2026", "nfl-weights-2026", "nfl-src-2026"],
    ):
        upload_dataset(path=path, destination=dest, name=dataset_ref)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp for model checkpoints",
    )
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Kaggle username for authentication",
    )
    parser.add_argument(
        "--train-data-dir",
        type=Path,
        help="Folder to which preprocessed data should be saved.",
        default=get_persistent_directory() / ".nfl-silver-solution" / "generated_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to folder containing training artifacts",
        default=get_persistent_directory()
        / ".nfl-silver-solution"
        / "training_artifacts",
    )
    args = parser.parse_args()
    main(args)

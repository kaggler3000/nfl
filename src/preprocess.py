from typing import List, Tuple

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from packaging.version import Version

from .config import Config
from .utils import (
    set_seed,
    load_input_output,
    standardize_dtypes,
    prepare_targets,
    standardize_play_direction,
)


class NflPlayerFrameDataset(Dataset):
    """
    Simple (X, y) dataset for per-frame per-player samples.
    X: standardized features
    y: displacement targets (e.g. dx, dy)
    """

    def __init__(self, X: np.ndarray, K: np.ndarray, y: np.ndarray, m: np.ndarray):
        assert X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X.astype(np.float32))
        self.K = torch.from_numpy(K.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.m = torch.from_numpy(m.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.X[idx],
            self.K[idx],
            self.y[idx],
            self.m[idx],
        )


class FeatureEngineer:
    def _height_to_numeric(self, st):
        if isinstance(st, str) and "-" in st:
            a, b = st.split("-")
            return float(a) + float(b) / 12.0
        return np.nan

    def player_based_features(self, df):

        # Player stats
        df["player_height_inches"] = df["player_height"] * 12.0
        df["bmi"] = (
            df["player_weight"] / (df["player_height_inches"] ** 2 + Config.EPS)
        ) * 703

        # Player roles
        df["is_offense"] = (df["player_side"] == "Offense").astype(bool)
        df["is_defense"] = (df["player_side"] == "Defense").astype(bool)
        df["is_receiver"] = (df["player_role"] == "Targeted Receiver").astype(bool)
        df["is_coverage"] = (df["player_role"] == "Defensive Coverage").astype(bool)
        df["is_passer"] = (df["player_role"] == "Passer").astype(bool)

        # Team side
        df["play_direction"] = (df["play_direction"] == "left").astype(bool)

        df["player_role_type"] = df["player_role"].map(
            Config.ROLES
        )  # apply index mapping to player roles
        df["player_role_type"] = df["player_role_type"].fillna(
            0
        )  # any NaNs get converted to 0

        return df

    def vector_interactions(self, a: pd.DataFrame, b: pd.DataFrame, prefix):
        """
        Input:
         - a: (N, 2)
         - b: (N, 2)

        Output:
         - result: (N, num_features) --> interactions between vector data `a` and `b`
        """
        a_x, a_y = a.values[:, 0], a.values[:, 1]
        b_x, b_y = b.values[:, 0], b.values[:, 1]
        result = pd.DataFrame(index=a.index)

        # Displacement
        result[f"{prefix}_diff_x"] = a_x - b_x
        result[f"{prefix}_diff_y"] = a_y - b_y
        diff_norm = (
            result[f"{prefix}_diff_x"] ** 2 + result[f"{prefix}_diff_y"] ** 2
        ) ** 0.5
        result[f"{prefix}_diff_norm"] = diff_norm

        # Angle
        a_norm = np.linalg.norm(a.values, axis=-1)  # (N,)
        b_norm = np.linalg.norm(b.values, axis=-1)
        result[f"{prefix}_cos_sim"] = (a_x * b_x + a_y * b_y) / (
            a_norm * b_norm + Config.EPS
        )  # cosine similarity
        result[f"{prefix}_angle"] = np.arccos(
            np.clip(result[f"{prefix}_cos_sim"], -1, 1)
        )

        # 2D cross product → scalar
        result[f"{prefix}_cross"] = a_x * b_y - a_y * b_x

        # projection of a onto direction of b
        result[f"{prefix}_proj_ab"] = (a_x * b_x + a_y * b_y) / (b_norm + Config.EPS)

        # Relative Velocity
        result[f"{prefix}_rv_x"] = (
            result[f"{prefix}_diff_x"] * Config.FPS / Config.horizon
        )
        result[f"{prefix}_rv_y"] = (
            result[f"{prefix}_diff_y"] * Config.FPS / Config.horizon
        )

        # Relative Acceleration
        result[f"{prefix}_ra_x"] = (
            result[f"{prefix}_rv_x"] * Config.FPS / Config.horizon
        )
        result[f"{prefix}_ra_y"] = (
            result[f"{prefix}_rv_y"] * Config.FPS / Config.horizon
        )

        # closing speed (rate distance increases or decreases)
        # positive = moving closer, negative = moving away
        result[f"{prefix}_closing_speed"] = result[f"{prefix}_rv_x"] * (
            result[f"{prefix}_diff_x"] / (diff_norm + Config.EPS)
        ) + result[f"{prefix}_rv_y"] * (
            result[f"{prefix}_diff_y"] / (diff_norm + Config.EPS)
        )

        return result

    def endpoint_interpolation(self, df):

        gcols = ["game_id", "play_id", "nfl_id"]

        if not {"velocity_x", "velocity_y"}.issubset(df.columns):
            raise ValueError(
                "endpoint_interpolation requires velocity_x/velocity_y. "
                "Run physics_based_features first."
            )

        last = (
            df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
            .groupby(gcols, as_index=False)
            .tail(1)
        )

        dt_total = Config.horizon / Config.FPS
        last["time_to_endpoint"] = last["frame_id"] / Config.FPS

        # Constant-velocity (linear) endpoint: x_T = x_0 + v * Δt
        last["endpoint_x_linear"] = last["x"] + last["velocity_x"] * dt_total
        last["endpoint_y_linear"] = last["y"] + last["velocity_y"] * dt_total
        # last['endpoint_x_linear'] = last['x'] + last['velocity_x'] * last['time_to_endpoint']
        # last['endpoint_y_linear'] = last['y'] + last['velocity_y'] * last['time_to_endpoint']

        # Constant-acceleration endpoint: x_T = x_0 + v * Δt + 0.5 * a * Δt^2
        if {"acceleration_x", "acceleration_y"}.issubset(df.columns):
            ax = last["acceleration_x"]
            ay = last["acceleration_y"]
        else:
            # If you haven't run acceleration features, just fall back to zero-accel
            ax = 0.0
            ay = 0.0

        last["endpoint_x_constacc"] = (
            last["x"] + last["velocity_x"] * dt_total + 0.5 * ax * (dt_total**2)
        )
        last["endpoint_y_constacc"] = (
            last["y"] + last["velocity_y"] * dt_total + 0.5 * ay * (dt_total**2)
        )
        # last['endpoint_x_constacc'] = (
        #     last['x'] + last['velocity_x'] * last['time_to_endpoint'] + 0.5 * ax * (last['time_to_endpoint'] ** 2)
        # )
        # last['endpoint_y_constacc'] = (
        #     last['y'] + last['velocity_y'] * last['time_to_endpoint'] + 0.5 * ay * (last['time_to_endpoint'] ** 2)
        # )

        # nfl-inspired interpolation
        last["endpoint_x_nfl"] = last["endpoint_x_linear"]
        last["endpoint_y_nfl"] = last["endpoint_y_linear"]
        receiver_mask = last["is_receiver"] == True
        coverage_mask = last["is_coverage"] == True
        last.loc[receiver_mask, "endpoint_x_nfl"] = last["ball_land_x"]
        last.loc[receiver_mask, "endpoint_y_nfl"] = last["ball_land_y"]
        last.loc[coverage_mask, "endpoint_x_nfl"] = (
            last.loc[coverage_mask, "ball_land_x"]
            + last.loc[coverage_mask, "wr_offset_x"]
        )
        last.loc[coverage_mask, "endpoint_y_nfl"] = (
            last.loc[coverage_mask, "ball_land_y"]
            + last.loc[coverage_mask, "wr_offset_y"]
        )

        for itp_type in ["linear", "constacc", "nfl"]:
            last[f"endpoint_x_{itp_type}"] = last[f"endpoint_x_{itp_type}"].clip(
                0, Config.FIELD_X_MAX
            )
            last[f"endpoint_y_{itp_type}"] = last[f"endpoint_y_{itp_type}"].clip(
                0, Config.FIELD_Y_MAX
            )
            b = last[[f"endpoint_x_{itp_type}", f"endpoint_y_{itp_type}"]]
            a = last[["x", "y"]]
            c = self.vector_interactions(a, b, prefix=f"{itp_type}_endpoint")
            last = pd.concat([last, c], axis=1)
            print(last.columns.tolist())

        # Keep only ids + new features and merge back onto all frames
        new_cols = [
            "endpoint_x_linear",
            "endpoint_y_linear",
            "endpoint_x_constacc",
            "endpoint_y_constacc",
            # Linear interpolation interactions
            "linear_endpoint_diff_x",
            "linear_endpoint_diff_y",
            "linear_endpoint_cos_sim",
            "linear_endpoint_angle",
            "linear_endpoint_rv_x",
            "linear_endpoint_rv_y",
            "linear_endpoint_ra_x",
            "linear_endpoint_ra_y",
            # Constacc interpolation interactions
            "constacc_endpoint_diff_x",
            "constacc_endpoint_diff_y",
            "constacc_endpoint_cos_sim",
            "constacc_endpoint_angle",
            "constacc_endpoint_rv_x",
            "constacc_endpoint_rv_y",
            "constacc_endpoint_ra_x",
            "constacc_endpoint_ra_y",
            # Newly added
            "endpoint_x_nfl",
            "endpoint_y_nfl",
            "nfl_endpoint_diff_x",
            "nfl_endpoint_diff_y",
            "nfl_endpoint_cos_sim",
            "nfl_endpoint_angle",
            "nfl_endpoint_rv_x",
            "nfl_endpoint_rv_y",
            "nfl_endpoint_ra_x",
            "nfl_endpoint_ra_y",
            "time_to_endpoint",
        ]
        endpoints = last[gcols + new_cols]

        df = df.merge(endpoints, on=gcols, how="left")

        df["linear_velocity_error_x"] = df["linear_endpoint_rv_x"] - df["velocity_x"]
        df["linear_velocity_error_y"] = df["linear_endpoint_rv_y"] - df["velocity_y"]
        df["constacc_velocity_error_x"] = (
            df["constacc_endpoint_rv_x"] - df["velocity_x"]
        )
        df["constacc_velocity_error_y"] = (
            df["constacc_endpoint_rv_y"] - df["velocity_y"]
        )
        df["nfl_velocity_error_x"] = df["nfl_endpoint_rv_x"] - df["velocity_x"]
        df["nfl_velocity_error_y"] = df["nfl_endpoint_rv_y"] - df["velocity_y"]

        df["linear_accel_error_x"] = df["linear_endpoint_ra_x"] - df["acceleration_x"]
        df["linear_accel_error_y"] = df["linear_endpoint_ra_y"] - df["acceleration_y"]
        df["constacc_accel_error_x"] = (
            df["constacc_endpoint_ra_x"] - df["acceleration_x"]
        )
        df["constacc_accel_error_y"] = (
            df["constacc_endpoint_ra_y"] - df["acceleration_y"]
        )
        df["nfl_accel_error_x"] = df["nfl_endpoint_ra_x"] - df["acceleration_x"]
        df["nfl_accel_error_y"] = df["nfl_endpoint_ra_y"] - df["acceleration_y"]

        return df

    def add_ball_features(self, df):

        a = df[["x", "y"]].values
        b = df[["ball_land_x", "ball_land_y"]].values  # (N, 2)

        df[f"ball_distance"] = np.sqrt(((a - b) ** 2).sum(axis=1))  # (N,)

        # Ball distance ranking
        # 1. collapse to last frame per player
        last_frame = df.groupby(["game_id", "play_id", "nfl_id"]).last().reset_index()
        # 2. compute rank within each play
        last_frame["ball_distance_rank"] = last_frame.groupby(["game_id", "play_id"])[
            "ball_distance"
        ].rank(method="dense", ascending=True)
        # 3. bring it back
        df = df.merge(
            last_frame[["game_id", "play_id", "nfl_id", "ball_distance_rank"]],
            on=["game_id", "play_id", "nfl_id"],
            how="left",
        )

        df[f"ball_dx"] = b[:, 0] - a[:, 0]
        df[f"ball_dy"] = b[:, 1] - a[:, 1]
        df[f"ball_angle"] = np.arctan2(df[f"ball_dy"], df[f"ball_dx"] + 1e-8)
        df[f"ball_diff_dir"] = np.rad2deg(df[f"ball_angle"]) - df["dir"]
        df[f"ball_diff_o"] = np.rad2deg(df[f"ball_angle"]) - df["o"]
        # df[f'ball_diff_dir'] = self.wrap_angle_deg(self.normalize_angle(np.rad2deg(df[f'ball_angle'])) - df['dir'])
        # df[f'ball_diff_o'] = self.wrap_angle_deg(self.normalize_angle(np.rad2deg(df[f'ball_angle'])) - df['o'])
        # projections
        df[f"ball_unit_dx"] = df[f"ball_dx"] / (df[f"ball_distance"] + Config.EPS)
        df[f"ball_unit_dy"] = df[f"ball_dy"] / (df[f"ball_distance"] + Config.EPS)
        df[f"ball_proj_mag"] = (a * b).sum(axis=1) / (
            (b * b).sum(axis=1) + Config.EPS
        )  # (N,)

        # Optional features
        df["closing_speed_ball"] = (
            df["velocity_x"] * df["ball_unit_dx"]
            + df["velocity_y"] * df["ball_unit_dy"]
        )
        df["velocity_toward_ball"] = df["velocity_x"] * np.cos(df["ball_angle"]) + df[
            "velocity_y"
        ] * np.sin(df["ball_angle"])
        df["velocity_alignment_ball"] = np.cos(np.deg2rad(df["ball_diff_dir"]))

        return df

    def player_interactions(self, df, K_NEIGH=3):

        out = []

        na_dict = {}
        for k in range(K_NEIGH):
            na_dict[f"distance_to_{k}_th"] = 50.0
            na_dict[f"dx_to_{k}_th"] = 50.0
            na_dict[f"dy_to_{k}_th"] = 50.0
            na_dict[f"v_dx_to_{k}_th"] = 0.0
            na_dict[f"v_dy_to_{k}_th"] = 0.0
            na_dict[f"a_dx_to_{k}_th"] = 0.0
            na_dict[f"a_dy_to_{k}_th"] = 0.0
            na_dict[f"v_closing_to_{k}_th"] = 0.0
            na_dict[f"v_to_{k}_th"] = 0.0
            na_dict[f"bearing_to_{k}_th"] = 0.0
            na_dict[f"bearing_diff_dir_{k}_th"] = 0.0
            na_dict[f"bearing_diff_o_{k}_th"] = 0.0
            na_dict[f"forward_sep_{k}_th"] = 50.0
            na_dict[f"lateral_sep_{k}_th"] = 50.0
        na_dict["num_neighbors_3"] = 0.0
        na_dict["num_neighbors_5"] = 0.0
        na_dict[f"wr_offset_x"] = 0.0
        na_dict[f"wr_offset_y"] = 0.0
        na_dict[f"wr_dist"] = 50.0
        na_dict[f"wr_rel_vx"] = 0.0
        na_dict[f"wr_rel_vy"] = 0.0
        na_dict[f"wr_bearing"] = 0.0

        # Loop only over plays (lightweight)
        for (gid, pid), group in tqdm(df.groupby(["game_id", "play_id"])):
            # one row per nfl_id (as in your original)
            play_df = group.groupby("nfl_id").last()

            nfl_ids = play_df.index.to_numpy()
            P = len(play_df)

            # Extract arrays
            pos = play_df[["x", "y"]].values  # (P, 2)
            sides = play_df["player_side"].values  # (P,)
            dirs = play_df["dir"].values
            ors = play_df["o"].values
            roles = play_df["player_role"].values
            v_x = play_df["velocity_x"].values  # (P,)
            v_y = play_df["velocity_y"].values  # (P,)
            v = np.stack([v_x, v_y], axis=-1)  # (P, 2)

            if P <= K_NEIGH:
                continue

            # Compute pairwise distance matrix (P × P)
            diff = pos[:, None, :] - pos[None, :, :]  # (P, P, 2)
            D = np.sqrt((diff**2).sum(axis=2))  # (P, P)

            # Mask same-team distances
            if Version(pd.__version__) >= Version("3.0"):
                sides = (
                    sides.to_numpy()
                )  # Modified 8/14: `sides` now defaults to StringArray after Pandas 3.0
            same_team = sides[:, None] == sides[None, :]
            D[same_team] = 50.0

            # For each player, find K nearest opponents
            # Use argpartition for efficiency
            nearest_idx = np.argpartition(D, K_NEIGH, axis=1)[:, :K_NEIGH]
            nearest_dists = np.take_along_axis(D, nearest_idx, axis=1)  # (P, K_NEIGH)
            nearest_dx = np.take_along_axis(diff[:, :, 0], nearest_idx, axis=1) / (
                nearest_dists + 1e-8
            )
            nearest_dy = np.take_along_axis(diff[:, :, 1], nearest_idx, axis=1) / (
                nearest_dists + 1e-8
            )

            # For each player, compute how many opponents within 3/5 yards (measure of density)
            num_3 = (D < 3).sum(axis=-1)  # (P,)
            num_5 = (D < 5).sum(axis=-1)  # (P,)

            # Compute closing speed
            u_x = (pos[:, None, 0] - pos[None, :, 0]) / (D + 1e-8)  # (P, P)
            u_y = (pos[:, None, 1] - pos[None, :, 1]) / (D + 1e-8)  # (P, P)
            rel_v_x = v_x[:, None] - v_x[None, :]  # (P, P)
            rel_v_y = v_y[:, None] - v_y[None, :]
            v_closing = -(rel_v_x * u_x + rel_v_y * u_y)
            nearest_v_closing = np.take_along_axis(v_closing, nearest_idx, axis=1)
            nearest_v_dx = np.take_along_axis(rel_v_x, nearest_idx, axis=1)
            nearest_v_dy = np.take_along_axis(rel_v_y, nearest_idx, axis=1)

            # Compute bearing
            p_x = pos[:, None, 0] - pos[None, :, 0]  # (P, P)
            p_y = pos[:, None, 1] - pos[None, :, 1]  # (P, P)
            bearing = np.arctan2(p_y, (p_x + 1e-8))  # (P, P)
            nearest_bearing = np.take_along_axis(bearing, nearest_idx, axis=1)

            # Forward separation
            u_v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)  # (P, 2)
            u_v_90 = np.stack([-u_v[:, 1], u_v[:, 0]], axis=-1)  # (P, 2)
            forward_sep = ((pos[:, None, :] - pos[None, :, :]) * u_v[:, None, :]).sum(
                axis=2
            )  # (P, P) dot product
            lateral_sep = (
                (pos[:, None, :] - pos[None, :, :]) * u_v_90[:, None, :]
            ).sum(axis=2)  # (P, P) dot product
            nearest_forward_sep = np.take_along_axis(forward_sep, nearest_idx, axis=1)
            nearest_lateral_sep = np.take_along_axis(lateral_sep, nearest_idx, axis=1)

            # Velocity towards ball
            # df['velocity_toward_ball'] = (
            #     df['velocity_x'] * np.cos(df['angle_to_ball']) +
            #     df['velocity_y'] * np.sin(df['angle_to_ball'])
            # )
            v_to_ball = v_x[:, None] * np.cos(bearing) + v_y[:, None] * np.sin(
                bearing
            )  # (P, P)
            nearest_v_to_ball = np.take_along_axis(v_to_ball, nearest_idx, axis=1)

            # WR mask
            wr_mask = (roles == "Targeted Receiver") | (roles == "Other Route Runner")
            if wr_mask.sum() > 0:  # if there even is a receiver
                wr_dist = D[:, wr_mask]
                nearest_wr = np.argpartition(wr_dist, kth=0, axis=1)[:, 0]
                wr_offset_x = np.take_along_axis(
                    diff[:, wr_mask, 0], nearest_wr[:, None], axis=1
                )
                wr_offset_y = np.take_along_axis(
                    diff[:, wr_mask, 1], nearest_wr[:, None], axis=1
                )
                wr_nearest_dists = np.take_along_axis(
                    wr_dist, nearest_wr[:, None], axis=1
                )
                wr_rel_vx = np.take_along_axis(
                    rel_v_x[:, wr_mask], nearest_wr[:, None], axis=1
                )
                wr_rel_vy = np.take_along_axis(
                    rel_v_y[:, wr_mask], nearest_wr[:, None], axis=1
                )

            # Emit rows
            for i, nid in enumerate(nfl_ids):
                row = {
                    "game_id": gid,
                    "play_id": pid,
                    "nfl_id": nid,
                }
                for k in range(K_NEIGH):
                    row[f"distance_to_{k}_th"] = nearest_dists[i, k]
                    row[f"dx_to_{k}_th"] = nearest_dx[i, k]
                    row[f"dy_to_{k}_th"] = nearest_dy[i, k]
                    row[f"v_dx_to_{k}_th"] = nearest_v_dx[i, k]
                    row[f"v_dy_to_{k}_th"] = nearest_v_dy[i, k]
                    row[f"v_closing_to_{k}_th"] = nearest_v_closing[i, k]
                    row[f"v_to_{k}_th"] = nearest_v_to_ball[i, k]
                    row[f"bearing_to_{k}_th"] = np.sin(nearest_bearing[i, k])
                    # row[f'bearing_diff_dir_{k}_th'] = (np.rad2deg(nearest_bearing[i, k]) - dirs[i])
                    # row[f'bearing_diff_o_{k}_th'] = (np.rad2deg(nearest_bearing[i, k]) - ors[i])
                    row[f"bearing_diff_dir_{k}_th"] = self.wrap_angle_deg(
                        self.normalize_angle(np.rad2deg(nearest_bearing[i, k]))
                        - dirs[i]
                    )
                    row[f"bearing_diff_o_{k}_th"] = self.wrap_angle_deg(
                        self.normalize_angle(np.rad2deg(nearest_bearing[i, k])) - ors[i]
                    )
                    row[f"forward_sep_{k}_th"] = nearest_forward_sep[i, k]
                    row[f"lateral_sep_{k}_th"] = nearest_lateral_sep[i, k]

                row[f"num_neighbors_3"] = num_3[i]
                row[f"num_neighbors_5"] = num_5[i]

                if wr_mask.sum() > 0:
                    row[f"wr_offset_x"] = wr_offset_x[i, 0]
                    row[f"wr_offset_y"] = wr_offset_y[i, 0]
                    row[f"wr_dist"] = wr_nearest_dists[i, 0]
                    row[f"wr_rel_vx"] = wr_rel_vx[i, 0]
                    row[f"wr_rel_vy"] = wr_rel_vy[i, 0]
                    row[f"wr_bearing"] = np.arctan2(
                        row[f"wr_offset_y"], row[f"wr_offset_x"] + 1e-8
                    )
                    # row[f'wr_bearing'] = self.normalize_angle(np.arctan2(row[f'wr_offset_y'], row[f'wr_offset_x'] + 1e-8))
                out.append(row)

        # Convert aggregated rows to DataFrame
        features = pd.DataFrame(out)

        # Force all expected columns to exist, even if `features` is empty
        all_cols = ["game_id", "play_id", "nfl_id", *na_dict.keys()]
        features = features.reindex(columns=all_cols)

        # Merge back to original df
        df = df.merge(features, on=["game_id", "play_id", "nfl_id"], how="left").fillna(
            na_dict
        )

        return df

    def physics_based_features(self, df):

        gcols = ["game_id", "play_id", "nfl_id"]

        # --- Convert direction to radians ---
        dir_rad = np.deg2rad(df["dir"].fillna(0))
        df["orientation_diff"] = np.abs(df["o"] - df["dir"])
        df["orientation_diff"] = np.minimum(
            df["orientation_diff"], 360 - df["orientation_diff"]
        )

        # --- Velocity components ---
        df["velocity_x"] = df["s"] * np.sin(dir_rad)
        df["velocity_y"] = df["s"] * np.cos(dir_rad)

        # --- Acceleration components ---
        df["acceleration_x"] = df.groupby(gcols)["velocity_x"].diff() * Config.FPS
        df["acceleration_y"] = df.groupby(gcols)["velocity_y"].diff() * Config.FPS
        df["accel_magnitude"] = np.sqrt(
            df["acceleration_x"] ** 2 + df["acceleration_y"] ** 2
        )
        # assert np.allclose(df['accel_magnitude'], df['a'].abs(), atol=1e-6) --> returns False, accel_magnitude is not equal to a

        # --- Physics: Momentum ---
        # p = m * v
        df["momentum_x"] = df["player_weight"] * df["velocity_x"]
        df["momentum_y"] = df["player_weight"] * df["velocity_y"]
        df["momentum_mag"] = np.sqrt(df["momentum_x"] ** 2 + df["momentum_y"] ** 2)

        # --- Physics: Kinetic Energy ---
        # KE = 1/2 * m * v^2
        df["speed_squared"] = df["s"] ** 2
        df["kinetic_energy"] = 0.5 * df["player_weight"] * df["speed_squared"]

        # Inertia
        df["inertia_momentum"] = df["momentum_mag"] / (df["accel_magnitude"] + 1e-6)

        # --- Frame-wise deltas (velocity change) ---
        # group by player trajectory (game_id, play_id, nfl_id)
        gcols = ["game_id", "play_id", "nfl_id"]

        df["velocity_x_delta"] = df.groupby(gcols)["velocity_x"].diff()
        df["velocity_y_delta"] = df.groupby(gcols)["velocity_y"].diff()
        df["speed_delta"] = df.groupby(gcols)["s"].diff()

        # --- Frame-wise deltas (acceleration change = jerk) ---
        # jerk = Δacceleration
        df["jerk_x"] = df.groupby(gcols)["acceleration_x"].diff() * Config.FPS
        df["jerk_y"] = df.groupby(gcols)["acceleration_y"].diff() * Config.FPS

        # --- Jerk magnitude ---
        df["jerk_mag"] = np.sqrt(df["jerk_x"] ** 2 + df["jerk_y"] ** 2)

        # --- EMA smoothing for noisy movement ---
        df["velocity_x_ema"] = df.groupby(gcols)["velocity_x"].transform(
            lambda x: x.ewm(alpha=0.3, adjust=False).mean()
        )
        df["velocity_y_ema"] = df.groupby(gcols)["velocity_y"].transform(
            lambda x: x.ewm(alpha=0.3, adjust=False).mean()
        )
        df["speed_ema"] = df.groupby(gcols)["s"].transform(
            lambda x: x.ewm(alpha=0.3, adjust=False).mean()
        )

        # ---- ball_land_x, ball_land_y ----
        df = self.add_ball_features(df)

        return df

    def team_based_features(self, df):

        team_xy = df.groupby(
            ["game_id", "play_id", "player_side", "frame_id"], as_index=False
        )[["x", "y"]].agg("mean")

        df = df.merge(
            team_xy.rename(columns={"x": "team_x", "y": "team_y"}),
            on=["game_id", "play_id", "player_side", "frame_id"],
            how="left",
        )

        df["team_centroid_dist"] = np.sqrt(
            (df["x"] - df["team_x"]) ** 2 + (df["y"] - df["team_y"]) ** 2
        )
        df["team_centroid_x"] = df["x"] - df["team_x"]
        df["team_centroid_y"] = df["y"] - df["team_y"]

        enemy_xy = team_xy.copy()
        enemy_xy["player_side"] = enemy_xy["player_side"].map(
            {"Offense": "Defense", "Defense": "Offense"}
        )
        enemy_xy = enemy_xy.rename(
            columns={
                "x": "enemy_x",
                "y": "enemy_y",
            }
        )
        df = df.merge(
            enemy_xy,
            on=["game_id", "play_id", "frame_id", "player_side"],
            how="left",
        )

        df["enemy_centroid_dist"] = (
            (df["x"] - df["enemy_x"]) ** 2 + (df["y"] - df["enemy_y"]) ** 2
        ) ** 0.5
        df["enemy_centroid_x"] = df["x"] - df["enemy_x"]
        df["enemy_centroid_y"] = df["y"] - df["enemy_y"]

        # Compute center of mass features
        com = (
            df.assign(
                mx=df["player_weight"] * df["x"],  # make new columns
                my=df["player_weight"] * df["y"],
            )
            .groupby(["game_id", "play_id", "player_side", "frame_id"])
            .agg(
                cm_x=("mx", "sum"),
                cm_y=("my", "sum"),
                total_mass=("player_weight", "sum"),
            )
            .reset_index()
        )
        com["x_com"] = com["cm_x"] / com["total_mass"]
        com["y_com"] = com["cm_y"] / com["total_mass"]
        com = com[["game_id", "play_id", "player_side", "frame_id", "x_com", "y_com"]]
        df = df.merge(
            com,
            on=["game_id", "play_id", "player_side", "frame_id"],
            how="left",
        )

        df["dist_com"] = (
            (df["x"] - df["x_com"]) ** 2 + (df["y"] - df["y_com"]) ** 2
        ) ** 0.5
        df["com_offset_x"] = df["x"] - df["x_com"]
        df["com_offset_y"] = df["y"] - df["y_com"]

        enemy_com = com.copy()
        enemy_com["player_side"] = enemy_com["player_side"].map(
            {
                "Offense": "Defense",
                "Defense": "Offense",
            }
        )
        enemy_com = enemy_com.rename(
            columns={
                "x_com": "enemy_x_com",
                "y_com": "enemy_y_com",
            }
        )

        df = df.merge(
            enemy_com,
            on=["game_id", "play_id", "player_side", "frame_id"],
            how="left",
        )

        df["dist_enemy_com"] = (
            (df["x"] - df["enemy_x_com"]) ** 2 + (df["y"] - df["enemy_y_com"]) ** 2
        ) ** 0.5
        df["enemy_com_offset_x"] = df["x"] - df["enemy_x_com"]
        df["enemy_com_offset_y"] = df["y"] - df["enemy_y_com"]

        df["dist_bw_com"] = (
            (df["x_com"] - df["enemy_x_com"]) ** 2
            + (df["y_com"] - df["enemy_y_com"]) ** 2
        ) ** 0.5
        df["bw_com_x"] = df["x_com"] - df["enemy_x_com"]
        df["bw_com_y"] = df["y_com"] - df["enemy_y_com"]

        return df

    def gnn_embeddings(self, df, tau=8.0, sigma=5.0):
        keys = ["game_id", "play_id", "nfl_id"]
        needed_cols = [
            "x",
            "y",
            "velocity_x",
            "velocity_y",
            "acceleration_x",
            "acceleration_y",
            "frame_id",
            "player_side",
        ] + keys

        src = df[needed_cols].copy()

        last = src.groupby(keys, as_index=False).tail(1)  # last frame
        neighbor = src.rename(
            columns={
                "x": "x_nb",
                "y": "y_nb",
                "velocity_x": "vx_nb",
                "velocity_y": "vy_nb",
                "acceleration_x": "ax_nb",
                "acceleration_y": "ay_nb",
                "nfl_id": "nfl_id_nb",
                "player_side": "player_side_nb",
            }
        )

        last = last.merge(  # for the last frame in each play, contains information between a pair of players
            neighbor,
            on=["game_id", "play_id", "frame_id"],
            how="left",
        )

        last = last[last.nfl_id != last.nfl_id_nb]  # removing self connections

        last["dx"] = last["x_nb"] - last["x"]
        last["dy"] = last["y_nb"] - last["y"]
        last["dvx"] = last["vx_nb"] - last["velocity_x"]
        last["dvy"] = last["vy_nb"] - last["velocity_y"]
        last["dax"] = last["ax_nb"] - last["velocity_x"]
        last["day"] = last["ay_nb"] - last["velocity_y"]
        last["dist"] = (last["dx"] ** 2 + last["dy"] ** 2) ** 0.5
        last["squared_dist"] = last["dx"] ** 2 + last["dy"] ** 2

        last["gnn_w"] = np.exp(-last["dist"] / tau)
        w_sum = last.groupby(["game_id", "play_id"])["gnn_w"].transform("sum")
        last["gnn_w_norm"] = last["gnn_w"] / (w_sum + 1e-8)

        last["is_ally"] = last.player_side == last.player_side_nb
        last["is_opp"] = ~last["is_ally"]

        last["gnn_ally_dx"] = last["dx"] * last["gnn_w_norm"] * last["is_ally"]
        last["gnn_opp_dx"] = last["dx"] * last["gnn_w_norm"] * last["is_opp"]
        last["gnn_ally_dy"] = last["dy"] * last["gnn_w_norm"] * last["is_ally"]
        last["gnn_opp_dy"] = last["dy"] * last["gnn_w_norm"] * last["is_opp"]
        last["gnn_ally_dvx"] = last["dvx"] * last["gnn_w_norm"] * last["is_ally"]
        last["gnn_opp_dvx"] = last["dvx"] * last["gnn_w_norm"] * last["is_opp"]
        last["gnn_ally_dvy"] = last["dvy"] * last["gnn_w_norm"] * last["is_ally"]
        last["gnn_opp_dvy"] = last["dvy"] * last["gnn_w_norm"] * last["is_opp"]
        last["gnn_ally_dist"] = last["dist"] * last["gnn_w_norm"] * last["is_ally"]
        last["gnn_opp_dist"] = last["dist"] * last["gnn_w_norm"] * last["is_opp"]
        last["ally_dist"] = np.where(
            last["is_ally"], last["dist"], np.nan
        )  # fill NaNs (so 'min' won't count it)
        last["opp_dist"] = np.where(last["is_opp"], last["dist"], np.nan)

        # RBF for measuring density?
        last["rbf_density"] = np.exp(-last["squared_dist"] / (2 * sigma**2))
        last["ally_rbf_density"] = last["rbf_density"] * last["is_ally"]
        last["opp_rbf_density"] = last["rbf_density"] * last["is_opp"]

        global_feats = last.groupby(
            ["game_id", "play_id", "nfl_id"], as_index=False
        ).agg(  # + >10 features
            num_allies=("is_ally", "sum"),
            num_opps=("is_opp", "sum"),
            gnn_ally_dx_sum=("gnn_ally_dx", "sum"),
            gnn_ally_dy_sum=("gnn_ally_dy", "sum"),
            gnn_opp_dx_sum=("gnn_opp_dx", "sum"),
            gnn_opp_dy_sum=("gnn_opp_dy", "sum"),
            gnn_ally_dvx_sum=("gnn_ally_dvx", "sum"),
            gnn_ally_dvy_sum=("gnn_ally_dvy", "sum"),
            gnn_opp_dvx_sum=("gnn_opp_dvx", "sum"),
            gnn_opp_dvy_sum=("gnn_opp_dvy", "sum"),
            gnn_ally_dist_sum=("gnn_ally_dist", "sum"),
            gnn_opp_dist_sum=("gnn_opp_dist", "sum"),
            gnn_ally_dist_min=("ally_dist", "min"),
            gnn_ally_dist_mean=("ally_dist", "mean"),
            gnn_opp_dist_min=("opp_dist", "min"),
            gnn_opp_dist_mean=("opp_dist", "mean"),
            gnn_ally_density=("ally_rbf_density", "sum"),
            gnn_opp_density=("opp_rbf_density", "sum"),
        )

        global_feats["gnn_opp_dist_min"] = np.log(
            global_feats["gnn_opp_dist_min"].values + 1
        )

        df = df.merge(
            global_feats,
            on=["game_id", "play_id", "nfl_id"],
            how="left",
        )

        return df

    def time_based_features(self, df):
        # print(f"Processing time-based features")
        df["num_frames_in"] = df.groupby(["game_id", "play_id", "nfl_id"])[
            "frame_id"
        ].transform("last")

        df["time_elapsed"] = df["num_frames_in"] / 10.0
        df["time_remaining"] = df["num_frames_output"] / 10.0
        df["progress_ratio"] = df["num_frames_in"] / (
            df["num_frames_in"] + df["num_frames_output"]
        )

        # next two frames
        df["next_x"] = df["x"] + df["velocity_x"] * 0.2
        df["next_y"] = df["y"] + df["velocity_y"] * 0.2
        df["next_ball_offset_x"] = df["ball_land_x"] - df["next_x"]
        df["next_ball_offset_y"] = df["ball_land_y"] - df["next_y"]
        df["next_ball_angle"] = np.arctan2(
            df["next_ball_offset_y"], (df["next_ball_offset_x"] + 1e-8)
        )
        df["next_ball_angle_x"] = np.sin(df["next_ball_angle"])
        df["next_ball_angle_y"] = np.cos(df["next_ball_angle"])

        # utilities
        df["time_squared"] = df["time_elapsed"] ** 2
        df["velocity_x_progress"] = df["velocity_x"] * df["progress_ratio"]
        df["velocity_y_progress"] = df["velocity_y"] * df["progress_ratio"]
        df["speed_scaled_by_time_left"] = df["s"] * df["time_remaining"]

        return df

    def wrap_angle_deg(self, a):
        return (a + 180) % 360 - 180

    def normalize_angle(self, a):  # normalize any arctan2's into the NFL convention
        return (90 - a) % 360

    def residual_features(self, df):
        residuals = []
        for (gid, pid, nid), group in tqdm(
            df.groupby(["game_id", "play_id", "nfl_id"])
        ):
            traj = group.copy()
            x = group["x"].values
            y = group["y"].values
            velocity_x = (x[-1] - x[0]) / (group["time_elapsed"].iloc[0] + 1e-8)
            velocity_y = (y[-1] - y[0]) / (group["time_elapsed"].iloc[0] + 1e-8)

            # location
            traj["expected_loc_x"] = (
                x[0] + (traj["frame_id"]) / 10.0 * velocity_x
            )  # better traj['frame_id'] than traj['frame_id'] - 1 (?)
            traj["expected_loc_y"] = y[0] + (traj["frame_id"]) / 10.0 * velocity_y
            traj["residual_loc_x"] = traj["expected_loc_x"] - traj["x"]
            traj["residual_loc_y"] = traj["expected_loc_y"] - traj["y"]

            residuals.append(traj)
        df_out = pd.concat(residuals, ignore_index=True)
        return df_out

    def __init__(self):
        self.f_dict = {
            "player-based": self.player_based_features,
            "physics-based": self.physics_based_features,
            "player-interactions": self.player_interactions,
            "team-based": self.team_based_features,
            "endpoint-based": self.endpoint_interpolation,
            "time-based": self.time_based_features,
            "residual-features": self.residual_features,
        }

    def transform(self, df):  # transform test data
        # Non-numeric features
        df["player_height"] = df["player_height"].apply(
            lambda x: self._height_to_numeric(x)
        )
        df["player_height"] = df["player_height"].astype("float64")

        # New features
        if "player-based" in Config.add_feature_groups:
            df = self.player_based_features(df)
        if "physics-based" in Config.add_feature_groups:
            df = self.physics_based_features(df)
        if "player-interactions" in Config.add_feature_groups:
            df = self.player_interactions(df)
        if "team-based" in Config.add_feature_groups:
            df = self.team_based_features(df)
        if "endpoint-based" in Config.add_feature_groups:
            df = self.endpoint_interpolation(df)
        if "time-based" in Config.add_feature_groups:
            df = self.time_based_features(df)
        if "residual-features" in Config.add_feature_groups:
            df = self.residual_features(df)
        if "gnn-features" in Config.add_feature_groups:
            df = self.gnn_embeddings(df)

        Config.binary_cols = []
        for col in df.columns:
            if df[col].dtype == "bool" and col in Config.feature_cols:
                Config.binary_cols.append(Config.feature_cols.index(col))
                df[col] = df[col].astype(int)

        return df


def get_play_level_sample(key, input_dict, output_dict, idx_x, idx_y):
    """
    Output:
     - sample_input: (num_players * num_frames, num_features)
     - sample_output: (num_players, num_frames_output, 2)
     - sample_p_mask: (num_players)
     - prediction_mask (for loss) -- computed later
     - key_padding_mask (for players) -- computed later
    """
    input_data = input_dict.get(key)
    if output_dict is not None:
        output_data = output_dict.get(key)
        output_groups = output_data.groupby(["nfl_id"], sort=False)

    player_groups = input_data.groupby("nfl_id", sort=False)
    num_players = len(player_groups)

    sample_input = []
    sample_output = []
    for nid, g in player_groups:
        input_window = g.tail(Config.window_size)

        if input_window.shape[0] < Config.window_size:
            pad_len = Config.window_size - input_window.shape[0]
            pad = pd.DataFrame(
                np.full((pad_len, input_window.shape[1]), np.nan),
                columns=input_window.columns,
            )
            input_window = pd.concat([pad, input_window], ignore_index=True)

        taken_features = Config.feature_cols
        # player_role_type feature is placed at the FRONT of the arr

        features = input_window[taken_features].to_numpy(dtype=np.float32)  # (F, f)
        col_means = np.nanmean(features, axis=0, keepdims=True)  # (1, f)
        nan_mask = np.isnan(features)
        if nan_mask.any():
            col_idx = np.where(nan_mask)[1]  # np.where(nan_mask) --> rows, cols
            features[nan_mask] = col_means[0, col_idx]
        player_input = np.nan_to_num(features, nan=0.0)
        sample_input.append(player_input)  # (frame_id, num_feats)

        if Config.TRAIN:
            if nid not in output_groups.groups:
                continue
            x_gt = output_groups.get_group((nid,))["x"].values
            y_gt = output_groups.get_group((nid,))["y"].values
            dx = x_gt - player_input[-1, idx_x]
            dy = y_gt - player_input[-1, idx_y]
            output_window = np.stack([dx, dy]).T

            sample_output.append(output_window)

    sample_input = np.concatenate(
        sample_input, axis=0
    )  # (num_players_predict * num_frames)

    if Config.SANITY_CHECKS:
        # Sanity Check: after sorting input_data by player_to_predict DESC
        player_order = input_data.drop_duplicates("nfl_id")
        flags = player_order["player_to_predict"].to_numpy()
        assert np.all(flags == np.sort(flags)[::-1]), (
            f"player_to_predict ordering is incorrect for play {key}: got {flags}"
        )

    if not Config.TRAIN:
        return sample_input, num_players
    else:
        sample_output = np.stack(sample_output, axis=0)
        return sample_input, sample_output, num_players


def stack_input(samples: list[np.ndarray], num_players: list[int]):
    """
    samples[i].shape = (P * F, f)
    Output:
     - X.shape = (B, P, F, f)
     - m.shape = (B, P) --> key_padding_mask
    """
    X = []
    m = []
    for i, s in tqdm(
        enumerate(samples), total=len(samples), desc="Collating input ..."
    ):
        D1, D2 = s.shape
        if Config.SANITY_CHECKS:
            assert D1 % num_players[i] == 0
        F = int(D1 / num_players[i])
        sample = s.reshape(
            num_players[i], F, D2
        )  # assuming adjacent frames belong to the same player
        if num_players[i] < Config.max_players:
            sample = np.concatenate(
                [sample, np.zeros(shape=(Config.max_players - num_players[i], F, D2))],
                axis=0,
            )

        sample_mask = np.zeros(shape=(Config.max_players))
        sample_mask[: num_players[i]] = 1.0
        X.append(sample)
        m.append(sample_mask)

    X = np.stack(X, axis=0)
    m = np.stack(m, axis=0)

    return X, m


def preprocess(input_df=None, output_df=None):
    # loading input/output
    if input_df is None:  # else, use the user-inputted dataframe
        input_df, output_df = load_input_output()  # concatenate w01 - w18

    # standardize dtypes
    input_df = standardize_dtypes(input_df)  # permutation invariant
    output_df = standardize_dtypes(output_df)  # permutation invariant

    if Config.FLIP_LEFT_RIGHT:
        input_df, output_df = standardize_play_direction(
            input_df, output_df
        )  # permutation invariant

    return input_df, output_df


def selective_transform(scaler, arr):
    return scaler.transform(arr)


import joblib


def load_and_preprocess(
    input_df, output_df, scaler_file_path=None, test_size=Config.test_size
) -> Tuple[DataLoader, DataLoader, List[str], StandardScaler]:
    """
    Load the training table, build train/val DataLoaders and fit a StandardScaler.

    Returns:
        train_loader: DataLoader
        val_loader: DataLoader
        feature_cols (list of feature names, in order)
        scaler (fitted StandardScaler instance)
    """

    # Feature Engineering
    feature_engineer = FeatureEngineer()
    processed_df = feature_engineer.transform(input_df)  # will drop unnecessary rows

    # Sort step
    processed_df = processed_df.sort_values(
        ["game_id", "play_id", "player_to_predict", "nfl_id", "frame_id"],
        ascending=[True, True, False, True, True],
    )
    output_df = output_df.sort_values(
        ["game_id", "play_id", "nfl_id", "frame_id"],
    )

    # Grouping features by unique plays
    input_dict = {
        (gid, pid): g
        for (gid, pid), g in processed_df.groupby(
            ["game_id", "play_id"],
            sort=False,  # already in sorted order (do not change it)
        )
    }
    output_dict = {
        (gid, pid): g
        for (gid, pid), g in output_df.groupby(
            ["game_id", "play_id"],
            sort=False,
        )
    }

    if not Config.TRAIN:  # if testing / submission
        indices = output_df[["game_id", "play_id"]].drop_duplicates()  # sorted index
        indices = [tuple(x) for x in indices.to_numpy()]
        idx_x = Config.feature_cols.index("x")
        idx_y = Config.feature_cols.index("y")
        plays = [
            get_play_level_sample(key, input_dict, None, idx_x, idx_y)
            for key in tqdm(indices)
        ]
        samples, num_players = zip(*plays)

        if Config.SANITY_CHECKS:
            assert scaler_file_path is not None

        scaler = joblib.load(scaler_file_path)

        X_test, K_test = stack_input(
            [selective_transform(scaler, arr) for arr in samples], num_players
        )

        return (
            X_test,  # the data
            K_test,  # masking players that don't need attention
            scaler,
        )

    # Train/Val split
    print(f"Generating Train/Val Split ... ")
    full_indices = output_df[["game_id", "play_id"]].drop_duplicates()  # DataFrame
    full_indices = [
        tuple(x) for x in full_indices.to_numpy()
    ]  # turn into list[tuple[int, int, int]]
    if test_size == 0.0:
        train_indices = full_indices
        val_indices = []
    else:
        if Config.split_type == "gkf":
            from sklearn.model_selection import KFold

            gkf = KFold(
                n_splits=5,
                # random_state=42,
            )

            X_dummy = np.zeros((len(full_indices), 1))  # GKF ignores data contents

            gkf_iterator = gkf.split(X_dummy)
            fold_indices = []
            for index, (train_idx, val_idx) in enumerate(gkf_iterator):
                fold_indices.append(
                    {
                        "train_idx": train_idx,
                        "val_idx": val_idx,
                    }
                )

            train_indices = full_indices
            val_indices = []

        else:
            train_indices, val_indices = train_test_split(
                full_indices,
                test_size=test_size,
                random_state=42,
            )

    # Loading training and validation data
    idx_x = Config.feature_cols.index("x")
    idx_y = Config.feature_cols.index("y")

    print(f"Getting play level samples ... ")
    train_plays = [
        get_play_level_sample(key, input_dict, output_dict, idx_x, idx_y)
        for key in tqdm(train_indices)
    ]
    val_plays = [
        get_play_level_sample(key, input_dict, output_dict, idx_x, idx_y)
        for key in tqdm(val_indices)
    ]

    # Loading and feature standardization
    xt, yt, num_players_t = list(zip(*train_plays))

    print(
        f"Fitting Standard Scaler ... "
    )  # fit on role is ok as long as you don't transform
    scaler = StandardScaler()
    scaler.fit(
        np.concatenate(xt, axis=0)
    )  # (tot, feature_number) --> fit on all concatenated training data

    X_train, K_train = stack_input(
        [selective_transform(scaler, arr) for arr in xt], num_players_t
    )
    y_train, m_train = prepare_targets(yt)

    if Config.split_type == "gkf":
        assert fold_indices is not None
        print(f"\n[INFO] GKF split has been created!")
        print(f"X_train.shape: {X_train.shape}")
        print(f"y_train.shape: {y_train.shape}")
        return (X_train, K_train, y_train, m_train, scaler, fold_indices)

    xv, yv, num_players_v = list(zip(*val_plays))
    X_val, K_val = stack_input(
        [selective_transform(scaler, arr) for arr in xv], num_players_v
    )
    y_val, m_val = prepare_targets(yv)

    print(f"\n[INFO] Train/Val split has been created!")
    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"X_val.shape: {X_val.shape}")
    print(f"y_val.shape: {y_val.shape}\n")

    return (X_train, K_train, y_train, m_train, X_val, K_val, y_val, m_val, scaler)


def get_dataloaders(
    X_train, K_train, y_train, m_train, X_val=None, K_val=None, y_val=None, m_val=None
):

    # Datasets + loaders
    train_ds = NflPlayerFrameDataset(X_train, K_train, y_train, m_train)

    if not Config.TRAIN:
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.batch_size,
            shuffle=False,  # False
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory,
            drop_last=False,
        )
        return train_loader

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        drop_last=False,
    )

    val_ds = NflPlayerFrameDataset(X_val, K_val, y_val, m_val)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader

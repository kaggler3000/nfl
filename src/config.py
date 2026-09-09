import torch


class Config:
    # File Paths
    BASE_DIR = None
    DATA_DIR = ""
    TRAIN_PATH = None
    TIME_TAG = None  # unique model checkpoint ID
    OUTPUT_DIR = None  # store model checkpoints
    SAVE_DIR = None
    TRAIN = False  # training flag
    DEBUG = False  # debugging flag
    SANITY_CHECKS = False  # debugging flag
    PREPROCESS = False  # data processing flag

    # Generated Data
    G_DATA_DIR = None  # store data checkpoints
    G_DATA_VERSION = 25  # data checkpoint version (int or string)

    # Constants
    EPS = 1e-8
    FPS = 10
    SECONDS_PER_FRAME = 0.1
    FIELD_X_MAX = 120.0  # max yardage across the long axis (X)
    FIELD_Y_MAX = 53.3  # max yardage across short axis (Y)
    ROLES = {
        # No valid role: 0,
        "Targeted Receiver": 1,
        "Defensive Coverage": 2,
        "Passer": 3,
        "Other Route Runner": 4,
    }

    # Data Processing
    FLIP_LEFT_RIGHT = True  # x --> FIELD_X_MAX - x
    FLIP_Y = True  # y --> FIELD_Y_MAX - y
    feature_cols = [
        # basic pack
        "x",
        "y",
        "s",
        "a",
        "dir",
        "o",
        "ball_land_x",
        "ball_land_y",
        "player_height",
        "player_weight",
        "absolute_yardline_number",
        "velocity_x",
        "velocity_y",
        "acceleration_x",
        "acceleration_y",
        "momentum_x",
        "momentum_y",
        "momentum_mag",
        "kinetic_energy",
        "inertia_momentum",
        "velocity_x_delta",
        "velocity_y_delta",
        "jerk_x",
        "jerk_y",
        "jerk_mag",
        "velocity_x_ema",
        "velocity_y_ema",
        "speed_ema",
        "bmi",
        # Endpoint-based features
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
        "play_direction",
        # player interactions
        "distance_to_0_th",
        "v_closing_to_0_th",
        "distance_to_1_th",
        "v_closing_to_1_th",
        "distance_to_2_th",
        "v_closing_to_2_th",
        "num_neighbors_3",
        "num_neighbors_5",
        "wr_offset_x",
        "wr_offset_y",
        "wr_dist",
        "wr_rel_vx",
        "wr_rel_vy",
        "bearing_to_0_th",
        "bearing_to_1_th",
        "bearing_to_2_th",
        # Ball features
        "ball_distance",
        "ball_distance_rank",
        "ball_dx",
        "ball_dy",
        "ball_angle",
        "ball_diff_dir",
        "ball_diff_o",
        "ball_unit_dx",
        "ball_unit_dy",
        "ball_proj_mag",
        "closing_speed_ball",
        "velocity_toward_ball",
        "velocity_alignment_ball",
        # time based features
        "num_frames_in",
        "time_elapsed",
        "time_remaining",
        "progress_ratio",
        "next_x",
        "next_y",
        "time_squared",
        "velocity_x_progress",
        "velocity_y_progress",
        "speed_scaled_by_time_left",
        # residual based features
        "expected_loc_x",
        "expected_loc_y",
        "residual_loc_x",
        "residual_loc_y",
        "num_allies",
        "num_opps",
        # GNN features
        "gnn_ally_dx_sum",
        "gnn_ally_dy_sum",
        "gnn_opp_dx_sum",
        "gnn_opp_dy_sum",
        "gnn_ally_dvx_sum",
        "gnn_ally_dvy_sum",
        "gnn_opp_dvx_sum",
        "gnn_opp_dvy_sum",
        "gnn_ally_dist_sum",
        "gnn_opp_dist_sum",
        "gnn_ally_dist_min",
        "gnn_ally_dist_mean",
        "gnn_opp_dist_min",
        "gnn_opp_dist_mean",
        "gnn_ally_density",
        "gnn_opp_density",
    ]  # keeps track of the features currently present in the DF
    target_cols = [
        "dx",
        "dy",
    ]
    add_feature_groups = [
        "player-based",
        "physics-based",
        "player-interactions",
        "team-based",
        "endpoint-based",
        "time-based",
        "residual-features",
        "gnn-features",
    ]

    # Training/Validation Split
    valid_frac = 0.1
    random_state = 42
    split_type = "gkf"  # or 'tts'
    all_folds = True  # True --> train on all folds

    # Model
    window_size = 10  # EDA
    horizon = 55
    max_players = 22  # a maximum of 22 players are on the field
    hidden_sizes = [128, 128]
    input_dim = len(feature_cols)
    output_dim = horizon * 2
    dropout = 0.1
    num_residual_layers = 2
    num_time_encoder_layers = 4
    num_player_encoder_layers = 4
    st_hidden_dim = 128
    mlp_hidden_dim = 256
    time_nheads = 8
    player_nheads = 8
    dim_feedforward = 512
    test_size = 0.25
    pooling_method = "last"  # 'mean', 'last+mean', 'att'

    # Loss
    loss_type = "huber"
    squared_weights = False
    time_decay = 0.06

    # Optimization
    batch_size = 64
    num_epochs = 50
    lr = 5e-4
    weight_decay = 1e-5
    clip_grad_norm = 5.0
    use_scheduler = False
    num_warmup_steps = 500
    use_ema = True

    # DataLoaders
    num_workers = 2
    pin_memory = True

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

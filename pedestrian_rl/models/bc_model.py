import torch
import torch.nn as nn
from ..utils.config_loader import load_config


class FiLMGenerator(nn.Module):
    '''
    Generate FiLM parameters from conditioning features.

    Output:
        gamma: multiplicative scale, initialized near identity
        beta : additive shift, initialized near zero
    '''
    def __init__(self, conditioning_dim: int, feature_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.LayerNorm(conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, feature_dim * 2),
        )

        # Start close to identity FiLM:
        # gamma ~= 1, beta ~= 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.feature_dim = feature_dim

    def forward(self, conditioning_feature):
        gamma_beta = self.net(conditioning_feature)
        gamma_raw, beta = torch.chunk(gamma_beta, 2, dim=-1)

        gamma = 1.0 + gamma_raw
        return gamma, beta


class BehaviorCloningPolicy(nn.Module):
    '''
    BC policy using BEV features + FiLM-conditioned scalar fusion.

    Inputs:
        bev_data         : (B, H, W, C) or (B, C, H, W)
        velocity_local   : (B, 2)
        speed            : (B,)
        yaw_sin          : (B,)
        yaw_cos          : (B,)
        goal_rel_local   : (B, 2)

    Outputs:
        pred_speed       : (B, 1)
        pred_direction   : (B, 2)

    Model Structures:
        LayerNorm(): ensure no data loss during different scales
        SiLU(): ensuure smooth actions output with all gradient
        Softplus(): ensure speed output is always > 0
    '''
    sim_config = load_config("sim_config.json")
    training_config = load_config("training_config.json")
    MAX_SPEED = sim_config["simulation"]["pedestrian"]["speed_range"][1]
    INPUT_CHANNELS = training_config["cnn"]["input_channels"]
    BEV_FEATURE_DIM = training_config["cnn"]["bev_feature_dim"]
    HIDDEN_DIM = training_config["cnn"]["hidden_dim"]
    DRIECTION_DIM = training_config["cnn"]["direction_dim"]
    STATE_FEATURE_DIM = training_config["cnn"]["state_feature_dim"]
    GOAL_FEATURE_DIM = training_config["cnn"]["goal_feature_dim"]
    DROPOUT = training_config["bc"]["params"]["dropout"]
    
    def __init__(
        self,
        cnn_encoder,
        bev_feature_dim=BEV_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        direction_dim=DRIECTION_DIM,
        dropout=DROPOUT,
        state_feature_dim=STATE_FEATURE_DIM,
        goal_feature_dim=GOAL_FEATURE_DIM,
        max_speed=MAX_SPEED
    ):
        super().__init__()

        self.cnn_encoder = cnn_encoder

        # velocity_local(2) + speed(1) + yaw_sin(1) + yaw_cos(1) = 5
        self.state_encoder = nn.Sequential(
            nn.Linear(5, state_feature_dim),
            nn.LayerNorm(state_feature_dim),
            nn.SiLU(),
            nn.Linear(state_feature_dim, state_feature_dim),
            nn.SiLU(),
        )

        # goal_rel_local(2)
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, goal_feature_dim),
            nn.LayerNorm(goal_feature_dim),
            nn.SiLU(),
            nn.Linear(goal_feature_dim, goal_feature_dim),
            nn.SiLU(),
        )

        conditioning_dim = state_feature_dim + goal_feature_dim
        self.film = FiLMGenerator(
            conditioning_dim=conditioning_dim,
            feature_dim=bev_feature_dim,
        )

        fusion_dim = bev_feature_dim + state_feature_dim + goal_feature_dim

        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.speed_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, direction_dim),
        )

        self.max_speed = float(max_speed)

    def forward(self, data):
        '''
        1. Get data
        2. Concatenate different data
        3. Send to trunk and different head
        4. return policy
        '''
        bev_data = data["bev_data"]
        velocity_local = data["velocity_local"]
        speed = data["speed"].unsqueeze(-1)
        yaw_sin = data["yaw_sin"].unsqueeze(-1)
        yaw_cos = data["yaw_cos"].unsqueeze(-1)
        goal_rel_local = data["goal_rel_local"]

        # ----- branch encoders -----
        bev_feature = self.cnn_encoder(bev_data)

        state_input = torch.cat(
            [velocity_local, speed, yaw_sin, yaw_cos],
            dim=-1,
        )
        state_feature = self.state_encoder(state_input)
        goal_feature = self.goal_encoder(goal_rel_local)

        # ----- FiLM conditioning on BEV -----
        conditioning_feature = torch.cat([state_feature, goal_feature], dim=-1)
        gamma, beta = self.film(conditioning_feature)
        bev_feature = gamma * bev_feature + beta

        # ----- fusion -----
        fused_feature = torch.cat(
            [bev_feature, state_feature, goal_feature],
            dim=-1,
        )
        latent_feature = self.trunk(fused_feature)

        pred_speed = self.speed_head(latent_feature) * self.max_speed
        pred_direction = self.direction_head(latent_feature)

        return {
            "pred_speed": pred_speed,
            "pred_direction": pred_direction,
        }
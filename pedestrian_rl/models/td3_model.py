import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_encoder import CNNEncoder
from ..utils.config_loader import load_config

sim_config = load_config("sim_config.json")
training_config = load_config("training_config.json")

MAX_SPEED = sim_config["simulation"]["pedestrian"]["speed_range"][1]

# cnn params
INPUT_CHANNELS = training_config["cnn"]["input_channels"]
BEV_FEATURE_DIM = training_config["cnn"]["bev_feature_dim"]
SCALAR_FEATURE_DIM = training_config["cnn"]["scalar_feature_dim"]
HIDDEN_DIM = training_config["cnn"]["hidden_dim"]
STATE_FEATURE_DIM = training_config["cnn"].get("state_feature_dim", 64)
GOAL_FEATURE_DIM = training_config["cnn"].get("goal_feature_dim", 64)

# td3 params
TD3_PARAMS = training_config["td3"]["params"]
ACTOR_LEARNING_RATE = TD3_PARAMS["actor_learning_rate"]
CRITIC_LEARNING_RATE = TD3_PARAMS["critic_learning_rate"]
ACTOR_WEIGHT_DECAY = TD3_PARAMS["actor_weight_decay"]
CRITIC_WEIGHT_DECAY = TD3_PARAMS["critic_weight_decay"]
GAMMA = TD3_PARAMS["gamma"]
TAU = TD3_PARAMS["tau"]
POLICY_NOISE = TD3_PARAMS["policy_noise"]
NOISE_CLIP = TD3_PARAMS["noise_clip"]
POLICY_DELAY = TD3_PARAMS["policy_delay"]
EXPLORATION_SPEED_NOISE = TD3_PARAMS["exploration_speed_noise"]
EXPLORATION_DIRECTION_NOISE = TD3_PARAMS["exploration_direction_noise"]
REPLAY_CAPACITY = TD3_PARAMS["replay_capacity"]
DROPOUT = TD3_PARAMS["dropout"]
ACTION_FEATURE_DIM = TD3_PARAMS.get("action_feature_dim", 32)


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

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, conditioning_feature):
        gamma_beta = self.net(conditioning_feature)
        gamma_raw, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = 1.0 + gamma_raw
        return gamma, beta


class Actor(nn.Module):
    '''
    TD3 actor with modality fusion:
        BEV encoder
        + state encoder
        + goal encoder
        + FiLM conditioning on BEV feature
    '''

    def __init__(
        self,
        cnn_encoder,
        bev_feature_dim=BEV_FEATURE_DIM,
        scalar_feature_dim=SCALAR_FEATURE_DIM,   # kept for compatibility
        hidden_dim=HIDDEN_DIM,
        max_speed=MAX_SPEED,
        dropout=DROPOUT,
        state_feature_dim=STATE_FEATURE_DIM,
        goal_feature_dim=GOAL_FEATURE_DIM,
    ):
        super().__init__()

        self.cnn_encoder = cnn_encoder
        self.max_speed = float(max_speed)

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

        self.film = FiLMGenerator(
            conditioning_dim=state_feature_dim + goal_feature_dim,
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
            nn.Linear(64, 2),
        )

    @staticmethod
    def normalize_direction(direction, eps=1e-6):
        norm = torch.norm(direction, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=eps)
        return direction / norm

    def forward(self, obs):
        bev_feature = self.cnn_encoder(obs["bev_data"])

        state_input = torch.cat(
            [
                obs["velocity_local"],
                obs["speed"].unsqueeze(-1),
                obs["yaw_sin"].unsqueeze(-1),
                obs["yaw_cos"].unsqueeze(-1),
            ],
            dim=-1,
        )
        state_feature = self.state_encoder(state_input)
        goal_feature = self.goal_encoder(obs["goal_rel_local"])

        conditioning_feature = torch.cat([state_feature, goal_feature], dim=-1)
        gamma, beta = self.film(conditioning_feature)
        bev_feature = gamma * bev_feature + beta

        fused_feature = torch.cat(
            [bev_feature, state_feature, goal_feature],
            dim=-1,
        )
        latent_feature = self.trunk(fused_feature)

        pred_speed = self.speed_head(latent_feature) * self.max_speed
        pred_direction = self.normalize_direction(self.direction_head(latent_feature))

        return torch.cat([pred_speed, pred_direction], dim=-1)


class CriticBranch(nn.Module):
    '''
    One critic branch Q(s, a) with modality fusion.
    '''

    def __init__(
        self,
        cnn_encoder,
        bev_feature_dim=BEV_FEATURE_DIM,
        scalar_feature_dim=SCALAR_FEATURE_DIM,   # kept for compatibility
        action_dim=3,
        hidden_dim=HIDDEN_DIM,
        max_speed=MAX_SPEED,
        dropout=DROPOUT,
        state_feature_dim=STATE_FEATURE_DIM,
        goal_feature_dim=GOAL_FEATURE_DIM,
        action_feature_dim=ACTION_FEATURE_DIM,
    ):
        super().__init__()

        self.cnn_encoder = cnn_encoder
        self.max_speed = float(max_speed)

        self.state_encoder = nn.Sequential(
            nn.Linear(5, state_feature_dim),
            nn.LayerNorm(state_feature_dim),
            nn.SiLU(),
            nn.Linear(state_feature_dim, state_feature_dim),
            nn.SiLU(),
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(2, goal_feature_dim),
            nn.LayerNorm(goal_feature_dim),
            nn.SiLU(),
            nn.Linear(goal_feature_dim, goal_feature_dim),
            nn.SiLU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_feature_dim),
            nn.LayerNorm(action_feature_dim),
            nn.SiLU(),
            nn.Linear(action_feature_dim, action_feature_dim),
            nn.SiLU(),
        )

        self.film = FiLMGenerator(
            conditioning_dim=state_feature_dim + goal_feature_dim,
            feature_dim=bev_feature_dim,
        )

        fusion_dim = (
            bev_feature_dim
            + state_feature_dim
            + goal_feature_dim
            + action_feature_dim
        )

        self.q_net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def normalize_action(self, action):
        speed = action[:, :1] / max(self.max_speed, 1e-6)
        direction = action[:, 1:3]
        return torch.cat([speed, direction], dim=-1)

    def forward(self, obs, action):
        bev_feature = self.cnn_encoder(obs["bev_data"])

        state_input = torch.cat(
            [
                obs["velocity_local"],
                obs["speed"].unsqueeze(-1),
                obs["yaw_sin"].unsqueeze(-1),
                obs["yaw_cos"].unsqueeze(-1),
            ],
            dim=-1,
        )
        state_feature = self.state_encoder(state_input)
        goal_feature = self.goal_encoder(obs["goal_rel_local"])

        conditioning_feature = torch.cat([state_feature, goal_feature], dim=-1)
        gamma, beta = self.film(conditioning_feature)
        bev_feature = gamma * bev_feature + beta

        action_feature = self.action_encoder(self.normalize_action(action))

        fused_feature = torch.cat(
            [bev_feature, state_feature, goal_feature, action_feature],
            dim=-1,
        )
        return self.q_net(fused_feature)


class TwinCritic(nn.Module):
    '''Twin critic used in TD3.'''

    def __init__(
        self,
        input_channels=INPUT_CHANNELS,
        bev_feature_dim=BEV_FEATURE_DIM,
        scalar_feature_dim=SCALAR_FEATURE_DIM,
        action_dim=3,
        hidden_dim=HIDDEN_DIM,
        max_speed=MAX_SPEED,
        dropout=DROPOUT,
        state_feature_dim=STATE_FEATURE_DIM,
        goal_feature_dim=GOAL_FEATURE_DIM,
        action_feature_dim=ACTION_FEATURE_DIM,
    ):
        super().__init__()

        self.q1 = CriticBranch(
            cnn_encoder=CNNEncoder(input_channels=input_channels, feature_dim=bev_feature_dim),
            bev_feature_dim=bev_feature_dim,
            scalar_feature_dim=scalar_feature_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_speed=max_speed,
            dropout=dropout,
            state_feature_dim=state_feature_dim,
            goal_feature_dim=goal_feature_dim,
            action_feature_dim=action_feature_dim,
        )
        self.q2 = CriticBranch(
            cnn_encoder=CNNEncoder(input_channels=input_channels, feature_dim=bev_feature_dim),
            bev_feature_dim=bev_feature_dim,
            scalar_feature_dim=scalar_feature_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_speed=max_speed,
            dropout=dropout,
            state_feature_dim=state_feature_dim,
            goal_feature_dim=goal_feature_dim,
            action_feature_dim=action_feature_dim,
        )

    def forward(self, obs, action):
        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)
        return q1, q2

    def q1_forward(self, obs, action):
        return self.q1(obs, action)


class ReplayBuffer:
    '''Replay buffer for TD3 training.'''

    def __init__(self, capacity=200000):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, obs, action, reward, next_obs, done):
        '''Store one transition.'''
        transition = {
            "obs": {
                "bev_data": np.asarray(obs["bev_data"], dtype=np.uint8),
                "velocity_local": np.asarray(obs["velocity_local"], dtype=np.float32),
                "goal_rel_local": np.asarray(obs["goal_rel_local"], dtype=np.float32),
                "yaw_sin": np.float32(obs["yaw_sin"]),
                "yaw_cos": np.float32(obs["yaw_cos"]),
                "speed": np.float32(obs["speed"]),
            },
            "action": np.asarray(action, dtype=np.float32),
            "reward": np.float32(reward),
            "next_obs": {
                "bev_data": np.asarray(next_obs["bev_data"], dtype=np.uint8),
                "velocity_local": np.asarray(next_obs["velocity_local"], dtype=np.float32),
                "goal_rel_local": np.asarray(next_obs["goal_rel_local"], dtype=np.float32),
                "yaw_sin": np.float32(next_obs["yaw_sin"]),
                "yaw_cos": np.float32(next_obs["yaw_cos"]),
                "speed": np.float32(next_obs["speed"]),
            },
            "done": np.float32(done),
        }
        self.buffer.append(transition)

    def sample(self, batch_size):
        '''Sample one mini-batch.'''
        batch = random.sample(self.buffer, batch_size)

        obs = {
            "bev_data": np.stack([item["obs"]["bev_data"] for item in batch], axis=0).astype(np.float32),
            "velocity_local": np.stack([item["obs"]["velocity_local"] for item in batch], axis=0),
            "goal_rel_local": np.stack([item["obs"]["goal_rel_local"] for item in batch], axis=0),
            "yaw_sin": np.asarray([item["obs"]["yaw_sin"] for item in batch], dtype=np.float32),
            "yaw_cos": np.asarray([item["obs"]["yaw_cos"] for item in batch], dtype=np.float32),
            "speed": np.asarray([item["obs"]["speed"] for item in batch], dtype=np.float32),
        }

        next_obs = {
            "bev_data": np.stack([item["next_obs"]["bev_data"] for item in batch], axis=0),
            "velocity_local": np.stack([item["next_obs"]["velocity_local"] for item in batch], axis=0).astype(np.float32),
            "goal_rel_local": np.stack([item["next_obs"]["goal_rel_local"] for item in batch], axis=0),
            "yaw_sin": np.asarray([item["next_obs"]["yaw_sin"] for item in batch], dtype=np.float32),
            "yaw_cos": np.asarray([item["next_obs"]["yaw_cos"] for item in batch], dtype=np.float32),
            "speed": np.asarray([item["next_obs"]["speed"] for item in batch], dtype=np.float32),
        }

        actions = np.stack([item["action"] for item in batch], axis=0).astype(np.float32)
        rewards = np.asarray([item["reward"] for item in batch], dtype=np.float32).reshape(-1, 1)
        dones = np.asarray([item["done"] for item in batch], dtype=np.float32).reshape(-1, 1)

        return obs, actions, rewards, next_obs, dones


class TD3Agent:
    '''
    TD3 agent for continuous pedestrian control.

    Action format:
        [speed, dir_right, dir_forward]
    '''

    def __init__(
        self,
        input_channels=INPUT_CHANNELS,
        bev_feature_dim=BEV_FEATURE_DIM,
        scalar_feature_dim=SCALAR_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        action_dim=3,
        max_speed=MAX_SPEED,
        actor_learning_rate=ACTOR_LEARNING_RATE,
        critic_learning_rate=CRITIC_LEARNING_RATE,
        actor_weight_decay=ACTOR_WEIGHT_DECAY,
        critic_weight_decay=CRITIC_WEIGHT_DECAY,
        gamma=GAMMA,
        tau=TAU,
        policy_noise=POLICY_NOISE,
        noise_clip=NOISE_CLIP,
        policy_delay=POLICY_DELAY,
        exploration_speed_noise=EXPLORATION_SPEED_NOISE,
        exploration_direction_noise=EXPLORATION_DIRECTION_NOISE,
        replay_capacity=REPLAY_CAPACITY,
        dropout=DROPOUT,
        device="cuda",
    ):
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.max_speed = float(max_speed)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = int(policy_delay)
        self.exploration_speed_noise = float(exploration_speed_noise)
        self.exploration_direction_noise = float(exploration_direction_noise)
        self.total_updates = 0

        self.actor = Actor(
            cnn_encoder=CNNEncoder(input_channels=input_channels, feature_dim=bev_feature_dim),
            bev_feature_dim=bev_feature_dim,
            scalar_feature_dim=scalar_feature_dim,
            hidden_dim=hidden_dim,
            max_speed=max_speed,
            dropout=dropout,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        self.critic = TwinCritic(
            input_channels=input_channels,
            bev_feature_dim=bev_feature_dim,
            scalar_feature_dim=scalar_feature_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_speed=max_speed,
            dropout=dropout,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=actor_learning_rate,
            weight_decay=actor_weight_decay,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=critic_learning_rate,
            weight_decay=critic_weight_decay,
        )

        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

    def move_obs_to_device(self, obs):
        '''Move one observation batch to device.'''
        return {
            "bev_data": torch.as_tensor(obs["bev_data"], dtype=torch.float32, device=self.device),
            "velocity_local": torch.as_tensor(obs["velocity_local"], dtype=torch.float32, device=self.device),
            "goal_rel_local": torch.as_tensor(obs["goal_rel_local"], dtype=torch.float32, device=self.device),
            "yaw_sin": torch.as_tensor(obs["yaw_sin"], dtype=torch.float32, device=self.device),
            "yaw_cos": torch.as_tensor(obs["yaw_cos"], dtype=torch.float32, device=self.device),
            "speed": torch.as_tensor(obs["speed"], dtype=torch.float32, device=self.device),
        }

    @staticmethod
    def normalize_direction_np(direction, eps=1e-6):
        '''Normalize local-frame 2D direction vectors.'''
        norm = float(np.linalg.norm(direction))
        if norm < eps:
            return np.array([0.0, 1.0], dtype=np.float32)
        return (direction / norm).astype(np.float32)

    def add_exploration_noise(self, action):
        '''Add exploration noise to one action.'''
        noisy_action = action.copy().astype(np.float32)
        noisy_action[0] += np.random.normal(0.0, self.exploration_speed_noise)
        noisy_action[1:3] += np.random.normal(0.0, self.exploration_direction_noise, size=2)

        noisy_action[0] = np.clip(noisy_action[0], 0.0, self.max_speed)
        noisy_action[1:3] = self.normalize_direction_np(noisy_action[1:3])
        return noisy_action

    def select_action(self, obs, add_noise=False):
        '''Select one action from the actor network.'''
        single_obs = {
            "bev_data": np.expand_dims(obs["bev_data"], axis=0),
            "velocity_local": np.expand_dims(obs["velocity_local"], axis=0),
            "goal_rel_local": np.expand_dims(obs["goal_rel_local"], axis=0),
            "yaw_sin": np.asarray([obs["yaw_sin"]], dtype=np.float32),
            "yaw_cos": np.asarray([obs["yaw_cos"]], dtype=np.float32),
            "speed": np.asarray([obs["speed"]], dtype=np.float32),
        }

        obs_tensor = self.move_obs_to_device(single_obs)
        with torch.no_grad():
            action = self.actor(obs_tensor)[0].detach().cpu().numpy().astype(np.float32)

        action[0] = np.clip(action[0], 0.0, self.max_speed)
        action[1:3] = self.normalize_direction_np(action[1:3])

        if add_noise:
            action = self.add_exploration_noise(action)

        return action

    def store_transition(self, obs, action, reward, next_obs, done):
        '''Store one transition in replay buffer.'''
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    @staticmethod
    def soft_update(source_model, target_model, tau):
        '''Soft update target network.'''
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def target_policy_smoothing(self, action):
        '''Apply TD3 target action smoothing.'''
        speed = action[:, :1]
        direction = action[:, 1:3]

        speed_noise = torch.randn_like(speed) * self.policy_noise
        direction_noise = torch.randn_like(direction) * self.policy_noise

        speed_noise = torch.clamp(speed_noise, -self.noise_clip, self.noise_clip)
        direction_noise = torch.clamp(direction_noise, -self.noise_clip, self.noise_clip)

        speed = torch.clamp(speed + speed_noise, 0.0, self.max_speed)
        direction = direction + direction_noise
        direction = F.normalize(direction, dim=-1)

        return torch.cat([speed, direction], dim=-1)

    def train_step(self, batch_size=128):
        '''Run one TD3 optimization step.'''
        if len(self.replay_buffer) < batch_size:
            return None

        self.total_updates += 1

        obs_np, actions_np, rewards_np, next_obs_np, dones_np = self.replay_buffer.sample(batch_size)
        obs = self.move_obs_to_device(obs_np)
        next_obs = self.move_obs_to_device(next_obs_np)
        actions = torch.as_tensor(actions_np, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            next_actions = self.target_policy_smoothing(next_actions)
            target_q1, target_q2 = self.critic_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            td_target = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = None
        if self.total_updates % self.policy_delay == 0:
            pred_actions = self.actor(obs)
            actor_loss = -self.critic.q1_forward(obs, pred_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.actor_target, self.tau)
            self.soft_update(self.critic, self.critic_target, self.tau)
            actor_loss_value = float(actor_loss.item())

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value,
            "buffer_size": len(self.replay_buffer),
            "total_updates": self.total_updates,
        }

    def save(self, save_path):
        '''Save one TD3 checkpoint.'''
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "total_updates": self.total_updates,
            "max_speed": self.max_speed,
        }
        torch.save(checkpoint, save_path)

    def load(self, checkpoint_path, load_optimizers=True):
        '''Load one TD3 checkpoint.'''
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])

        if load_optimizers:
            if "actor_optimizer_state_dict" in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            if "critic_optimizer_state_dict" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        self.total_updates = int(checkpoint.get("total_updates", 0))
        return checkpoint

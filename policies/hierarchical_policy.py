"""
Hierarchical Path Planning Policy 2.0
- Uses FusionFeaturesExtractor (CNN + Physics)
- Kinematic Anchors based on Turn Bias + Current Steering
- Predicts Waypoints (Planning Layer) as deviations from curved path
- Outputs Controls (Steering/Throttle/Brake)

Changes:
- Anchors are now dynamic, based on turn command and vehicle state
- Blends navigation intent with steering continuity
- Progressive curvature for distant waypoints
- Supports continuous turn bias [-1, 1] instead of discrete {-1, 0, 1}
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Type, Any, Union
from gymnasium import spaces
import numpy as np

from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import Distribution


class HierarchicalPathPlanningPolicy(RecurrentActorCriticPolicy):
    """
    Hierarchical policy:
    1. Fusion Extractor (Pixels + Turn Bias + Physics) -> Latent
    2. LSTM -> Memory
    3. Kinematic Anchors (from Turn Bias + Steering) -> Curved base path
    4. Planning Head -> Deviations from kinematic path
    5. Control Head -> Action (Steer, Thr, Brk)
    """

    # Normalization constant for waypoints (meters -> normalized space)
    WAYPOINT_NORM_SCALE = 20.0

    # Telemetry vector indices (Protocol v2.0)
    IDX_TURN_BIAS = 0      # Continuous turn command [-1, 1]
    IDX_RESERVED = 1       # Reserved (always 0)
    IDX_GOAL_DIST = 2      # Goal distance (masked)
    IDX_SPEED = 3          # Vehicle speed (m/s)
    IDX_YAW_RATE = 4       # Yaw rate (rad/s)
    IDX_LAST_STEER = 5     # Previous steering command
    IDX_LAST_THR = 6       # Previous throttle command
    IDX_LAST_BRK = 7       # Previous brake command
    IDX_LAT_ERR = 8        # Lateral error from path
    IDX_HDG_ERR = 9        # Heading error from path
    IDX_KAPPA = 10         # Path curvature
    IDX_DS = 11            # Distance traveled

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = None,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            lstm_hidden_size: int = 256,
            n_lstm_layers: int = 1,
            shared_lstm: bool = False,
            enable_critic_lstm: bool = True,
            lstm_kwargs: Optional[Dict[str, Any]] = None,
            # --- HIERARCHICAL ARGS ---
            num_waypoints: int = 5,
            waypoint_horizon: float = 2.5,
            repulsion_weight: float = 2.0,
            waypoint_loss_weight: float = 0.15,
            # --- KINEMATIC ANCHOR ARGS (v2.0) ---
            use_kinematic_anchors: bool = True,
            curvature_gain: float = 0.18,
            command_blend_factor: float = 0.6,
            steering_blend_factor: float = 0.4,
            progressive_curvature_exp: float = 1.15,
            max_deviation_meters: float = 8.0,
    ):
        self.num_waypoints = num_waypoints
        self.waypoint_horizon = waypoint_horizon
        self.repulsion_weight = repulsion_weight
        self.waypoint_loss_weight = waypoint_loss_weight
        self.waypoint_dim = num_waypoints * 2
        self.planning_hidden_dim = 256
        self.control_hidden_dim = 128

        # Kinematic anchor parameters
        self.use_kinematic_anchors = use_kinematic_anchors
        self.curvature_gain = curvature_gain
        self.command_blend_factor = command_blend_factor
        self.steering_blend_factor = steering_blend_factor
        self.progressive_curvature_exp = progressive_curvature_exp
        self.max_deviation_meters = max_deviation_meters
        self.waypoint_spacing = 4.0  # meters between waypoints

        if net_arch is None:
            net_arch = dict(pi=[64], vf=[64])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )

        # Fix LSTM input dimension if needed
        if self.lstm_actor.input_size != self.features_dim:
            self.lstm_actor = nn.LSTM(
                self.features_dim,
                self.lstm_hidden_size,
                num_layers=self.n_lstm_layers,
                batch_first=False
            )

        if self.enable_critic_lstm and self.lstm_critic.input_size != self.features_dim:
            self.lstm_critic = nn.LSTM(
                self.features_dim,
                self.lstm_hidden_size,
                num_layers=self.n_lstm_layers,
                batch_first=False
            )

        self._build_hierarchical_heads()

        # Buffer for straight-line anchors (fallback when kinematic disabled)
        self.register_buffer('static_anchors', self._create_static_anchors())

        # State tracking
        self.last_waypoints = None
        self.last_anchors = None

    def _create_static_anchors(self) -> torch.Tensor:
        """Create straight-line anchors (0, dist) in car frame."""
        anchors = torch.zeros(1, self.num_waypoints, 2)
        for i in range(self.num_waypoints):
            anchors[0, i, 0] = 0.0  # X=0 (Center)
            anchors[0, i, 1] = (i + 1) * self.waypoint_spacing  # Y+ (Forward)
        return anchors

    def _build_hierarchical_heads(self) -> None:
        if self.mlp_extractor is not None:
            head_input_dim = self.mlp_extractor.latent_dim_pi
        else:
            head_input_dim = self.lstm_output_dim

        # 1. Planning Head (Predicts deviations from anchors)
        self.planning_head = nn.Sequential(
            nn.Linear(head_input_dim, self.planning_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.planning_hidden_dim, self.planning_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.planning_hidden_dim // 2, self.waypoint_dim)
        )

        # Init planning head to near-zero (start with anchor paths)
        with torch.no_grad():
            self.planning_head[-1].weight.mul_(0.01)
            self.planning_head[-1].bias.fill_(0.0)

        # 2. Control Head (Inputs: LSTM Features + Predicted Waypoints)
        control_input_dim = head_input_dim + self.waypoint_dim
        self.control_head = nn.Sequential(
            nn.Linear(control_input_dim, self.control_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.control_hidden_dim, self.control_hidden_dim // 2),
            nn.ReLU(),
        )

        # 3. Final Action Projection
        self.action_net = nn.Linear(self.control_hidden_dim // 2, self.action_space.shape[0])

        # Value Net (Standard)
        if self.mlp_extractor is None:
            self.value_net = nn.Linear(self.lstm_output_dim, 1)

    def _compute_kinematic_anchors(self, obs_vec: torch.Tensor) -> torch.Tensor:
        """
        Compute curved anchor points based on turn command and current steering.

        This creates an inductive bias toward continuing the current maneuver
        while respecting the high-level navigation intent.

        Args:
            obs_vec: (batch_size, 12) telemetry vector

        Returns:
            anchors: (batch_size, num_waypoints, 2) anchor positions in vehicle frame
        """
        batch_size = obs_vec.shape[0]
        device = obs_vec.device

        # Extract relevant signals
        turn_bias = obs_vec[:, self.IDX_TURN_BIAS]      # [-1, 1] navigation command
        last_steer = obs_vec[:, self.IDX_LAST_STEER]    # [-1, 1] current steering

        # Optional: Use heading error for additional context
        hdg_err = obs_vec[:, self.IDX_HDG_ERR]          # radians, path-relative

        # Blend navigation command with steering continuity
        # High command_blend = trust navigation intent
        # High steering_blend = trust current steering (smooth continuity)
        #
        # When turn_bias is strong (|bias| > 0.5), trust it more
        # When turn_bias is weak, trust current steering for smooth driving
        command_strength = torch.abs(turn_bias)

        # Adaptive blending based on command strength
        # Strong command -> follow it; Weak command -> smooth driving
        adaptive_command_weight = self.command_blend_factor + 0.3 * command_strength
        adaptive_steer_weight = 1.0 - adaptive_command_weight

        # Compute effective curvature signal
        # Negative turn_bias = turn left (negative X in vehicle frame)
        # Positive turn_bias = turn right (positive X in vehicle frame)
        effective_curvature = (
            adaptive_command_weight * turn_bias +
            adaptive_steer_weight * last_steer
        )
        effective_curvature = torch.clamp(effective_curvature, -1.0, 1.0)

        # Generate curved anchors
        anchors = torch.zeros(batch_size, self.num_waypoints, 2, device=device)

        for i in range(self.num_waypoints):
            dist = (i + 1) * self.waypoint_spacing

            # Progressive curvature: curves more at distance
            # This helps with sharper turns at planning horizon
            waypoint_index = i + 1
            progressive_factor = waypoint_index ** self.progressive_curvature_exp

            # Angle for this waypoint
            # angle = curvature_signal * gain * progressive_factor
            angle = effective_curvature * self.curvature_gain * progressive_factor

            # Convert to Cartesian coordinates in vehicle frame
            # X = lateral (positive = right), Y = forward (positive = ahead)
            anchors[:, i, 0] = dist * torch.sin(angle)  # Lateral
            anchors[:, i, 1] = dist * torch.cos(angle)  # Forward

        return anchors

    def _compute_waypoints(self, latent_pi: torch.Tensor, obs_vec: torch.Tensor) -> torch.Tensor:
        """
        Compute final waypoints as anchors + learned deviations.

        Args:
            latent_pi: (batch_size, latent_dim) LSTM output
            obs_vec: (batch_size, 12) telemetry vector

        Returns:
            waypoints: (batch_size, num_waypoints, 2) waypoint positions
        """
        batch_size = latent_pi.shape[0]

        # 1. Get anchors (kinematic or static)
        if self.use_kinematic_anchors:
            anchors = self._compute_kinematic_anchors(obs_vec)
        else:
            # Expand static anchors to batch size
            anchors = self.static_anchors.expand(batch_size, -1, -1).clone()

        self.last_anchors = anchors  # Save for visualization/debugging

        # 2. Predict deviations from anchors
        deviations = self.planning_head(latent_pi).reshape(-1, self.num_waypoints, 2)

        # Constrain deviations (reduced range since anchors handle more)
        deviations = torch.tanh(deviations) * self.max_deviation_meters

        # 3. Final waypoints = anchors + corrections
        waypoints = anchors + deviations

        return waypoints

    def _compute_control_features(self, latent_pi: torch.Tensor, waypoints: torch.Tensor) -> torch.Tensor:
        """Fuse LSTM memory with planned waypoints for control."""
        # Flatten and normalize waypoints
        wp_flat = waypoints.reshape(-1, self.waypoint_dim)
        wp_norm = wp_flat / self.WAYPOINT_NORM_SCALE

        # Concatenate memory + plan
        return self.control_head(torch.cat([latent_pi, wp_norm], dim=-1))

    def forward(self, obs, lstm_states, episode_starts, deterministic=False):
        """Forward pass through the hierarchical policy."""
        features = self.extract_features(obs)

        # Standard SB3 Recurrent logic
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi, lstm_states_pi = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )

        if self.enable_critic_lstm:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        else:
            latent_vf = vf_features
            lstm_states_vf = lstm_states.vf

        if self.mlp_extractor is not None:
            latent_pi = self.mlp_extractor.forward_actor(latent_pi)
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # --- Hierarchical Planning Step ---
        # Extract physics vector for kinematic anchors
        obs_vec = obs['vec'] if isinstance(obs, dict) else obs

        waypoints = self._compute_waypoints(latent_pi, obs_vec)
        self.last_waypoints = waypoints  # Save for logging/visualization

        control_features = self._compute_control_features(latent_pi, waypoints)

        # Distribution & Action
        distribution = self._get_action_dist_from_latent(control_features)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, values, log_prob, RNNStates(pi=lstm_states_pi, vf=lstm_states_vf)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        """Evaluate actions for PPO update."""
        features = self.extract_features(obs)

        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi, _ = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )

        if self.enable_critic_lstm:
            latent_vf, _ = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        else:
            latent_vf = vf_features

        if self.mlp_extractor is not None:
            latent_pi = self.mlp_extractor.forward_actor(latent_pi)
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Extract physics vector
        obs_vec = obs['vec'] if isinstance(obs, dict) else obs

        # Plan -> Control
        waypoints = self._compute_waypoints(latent_pi, obs_vec)
        control_features = self._compute_control_features(latent_pi, waypoints)

        distribution = self._get_action_dist_from_latent(control_features)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)

        return values, log_prob, entropy

    def get_distribution(self, obs, lstm_states, episode_starts):
        """Get action distribution (for deterministic action selection)."""
        features = self.extract_features(obs)
        latent_pi, lstm_states_pi = self._process_sequence(
            features, lstm_states.pi, episode_starts, self.lstm_actor
        )

        if self.mlp_extractor is not None:
            latent_pi = self.mlp_extractor.forward_actor(latent_pi)

        # Extract physics vector
        obs_vec = obs['vec'] if isinstance(obs, dict) else obs

        waypoints = self._compute_waypoints(latent_pi, obs_vec)
        self.last_waypoints = waypoints

        control_features = self._compute_control_features(latent_pi, waypoints)

        return (
            self._get_action_dist_from_latent(control_features),
            RNNStates(pi=lstm_states_pi, vf=lstm_states.vf)
        )

    def predict_values(self, obs, lstm_states, episode_starts):
        """
        Predict values for given observations.

        Note: SB3-contrib calls this with lstm_states already being the vf component
        (a tuple), not the full RNNStates object. So we use lstm_states directly.
        """
        features = self.extract_features(obs)

        if self.share_features_extractor:
            vf_features = features
        else:
            _, vf_features = features

        if self.enable_critic_lstm:
            # lstm_states is already the vf states tuple (not RNNStates object)
            latent_vf, _ = self._process_sequence(
                vf_features, lstm_states, episode_starts, self.lstm_critic
            )
        else:
            latent_vf = vf_features

        if self.mlp_extractor is not None:
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        return self.value_net(latent_vf)

    def get_waypoints_for_visualization(self) -> Optional[np.ndarray]:
        """
        Get the last predicted waypoints for visualization.

        Returns:
            waypoints: (num_waypoints, 2) numpy array or None
        """
        if self.last_waypoints is None:
            return None

        # Detach and convert to numpy
        wp = self.last_waypoints.detach()
        if wp.dim() == 3:
            wp = wp[0]  # Take first batch element
        return wp.cpu().numpy()

    def get_anchors_for_visualization(self) -> Optional[np.ndarray]:
        """
        Get the last computed anchors for visualization.

        Returns:
            anchors: (num_waypoints, 2) numpy array or None
        """
        if self.last_anchors is None:
            return None

        anchors = self.last_anchors.detach()
        if anchors.dim() == 3:
            anchors = anchors[0]
        return anchors.cpu().numpy()
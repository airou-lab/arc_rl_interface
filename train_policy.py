"""
Enhanced Training Script for Unity Camera Environment
(Strictly Passive: RGB-only, no semantic segmentation, no action blending)
"""

import os
import sys
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch.nn as nn
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from gymnasium.wrappers import TimeLimit

# Ensure unity_env module is accessible
sys.path.append(os.path.dirname(__file__))
from unity_camera_env import UnityCameraEnv

# ========================= Custom Callbacks ========================= #

class EnhancedLoggingCallback(BaseCallback):
    """Enhanced callback for detailed training metrics"""
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.episode_rewards = []
        self.episode_lengths = []

        # Create metrics file
        self.metrics_file = self.log_dir / "training_metrics.csv"
        with open(self.metrics_file, 'w') as f:
            f.write("timestep,episode,reward,length,lane_confidence,coverage,deviation\n")

    def _on_step(self) -> bool:
        # Log detailed metrics from environment (first env only)
        try:
            if hasattr(self.training_env, 'envs') and self.training_env.envs:
                env = self.training_env.envs[0]
                if hasattr(env, 'unwrapped'):
                    unity_env = env.unwrapped
                    info = unity_env._get_info() if hasattr(unity_env, '_get_info') else {}
                    with open(self.metrics_file, 'a') as f:
                        f.write(f"{self.num_timesteps},"
                                f"{len(self.episode_rewards)},"
                                f"{self.episode_rewards[-1] if self.episode_rewards else 0},"
                                f"{self.episode_lengths[-1] if self.episode_lengths else 0},"
                                f"{info.get('confidence', 0)},"
                                f"{info.get('coverage', 0)},"
                                f"{info.get('deviation', 0)}\n")
        except Exception:
            # Keep training even if logging fails
            pass
        return True

    def _on_rollout_end(self) -> None:
        # Log episode statistics if available
        try:
            dones = self.locals.get('dones', None)
            rewards = self.locals.get('rewards', None)
            if dones is not None and rewards is not None and np.any(dones):
                episode_reward = float(np.sum(rewards))
                # episode length isn't trivial to extract reliably here; omit or set to 0
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(0)
                if self.verbose > 0:
                    print(f"Episode {len(self.episode_rewards)}: Reward = {episode_reward:.2f}")
        except Exception:
            pass


class AdaptiveCurriculumCallback(BaseCallback):
    """Implements curriculum learning by adjusting environment difficulty"""
    def __init__(self,
                 initial_max_steps: int = 100,
                 target_max_steps: int = 500,
                 performance_threshold: float = 0.7,
                 window_size: int = 10,
                 verbose: int = 0):
        super().__init__(verbose)
        self.initial_max_steps = initial_max_steps
        self.target_max_steps = target_max_steps
        self.current_max_steps = initial_max_steps
        self.performance_threshold = performance_threshold
        self.window_size = window_size
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Check if episode ended
        try:
            dones = self.locals.get('dones', None)
            rewards = self.locals.get('rewards', None)
            if dones is not None and rewards is not None and np.any(dones):
                # Record episode reward
                episode_reward = float(np.sum(rewards))
                self.episode_rewards.append(episode_reward)

                # Check if we should increase difficulty
                if len(self.episode_rewards) >= self.window_size:
                    recent_rewards = self.episode_rewards[-self.window_size:]
                    avg_performance = float(np.mean(recent_rewards))

                    # Increase difficulty if performance is good
                    if avg_performance > self.performance_threshold:
                        self.current_max_steps = min(
                            self.current_max_steps + 50,
                            self.target_max_steps
                        )

                        # Update environment(s)
                        if hasattr(self.training_env, 'envs'):
                            for env in self.training_env.envs:
                                if hasattr(env, 'unwrapped'):
                                    env.unwrapped.max_steps = self.current_max_steps

                        if self.verbose > 0:
                            print(f"Curriculum: Increased max_steps to {self.current_max_steps}")
        except Exception:
            pass
        return True


# ========================= Training Configuration ========================= #

class TrainingConfig:
    """Configuration class for training parameters"""
    def __init__(self, **kwargs):
        # Environment settings
        self.capture_dir = kwargs.get('capture_dir', '../../Assets/Captures')
        self.img_size = kwargs.get('img_size', (84, 84))
        self.max_steps = kwargs.get('max_steps', None)

        # Enhanced features (passive by default: flow/path disabled)
        self.use_enhanced_detection = kwargs.get('use_enhanced_detection', True)
        self.use_optical_flow      = kwargs.get('use_optical_flow', False)
        self.use_path_planning     = kwargs.get('use_path_planning', False)
        self.use_augmentation      = kwargs.get('use_augmentation', True)

        # PPO hyperparameters
        self.learning_rate = kwargs.get('learning_rate', 2.5e-4)
        self.n_steps = kwargs.get('n_steps', 512)
        self.batch_size = kwargs.get('batch_size', 64)
        self.n_epochs = kwargs.get('n_epochs', 10)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.clip_range = kwargs.get('clip_range', 0.2)
        self.ent_coef = kwargs.get('ent_coef', 0.01)
        self.vf_coef = kwargs.get('vf_coef', 0.5)
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)

        # Training settings
        self.total_timesteps = kwargs.get('total_timesteps', 100_000)
        self.save_freq = kwargs.get('save_freq', 10_000)
        self.eval_freq = kwargs.get('eval_freq', 5_000)
        self.eval_episodes = kwargs.get('eval_episodes', 5)

        # Curriculum learning
        self.use_curriculum = kwargs.get('use_curriculum', False)
        self.curriculum_initial_steps = kwargs.get('curriculum_initial_steps', 100)
        self.curriculum_target_steps = kwargs.get('curriculum_target_steps', 500)

        # Logging
        self.tensorboard_log = kwargs.get('tensorboard_log', './tensorboard_logs')
        self.verbose = kwargs.get('verbose', 1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# ========================= Environment Factory ========================= #

def create_environment(config: TrainingConfig) -> gym.Env:
    """
    Create Unity camera environment with specified configuration
    """
    env_kwargs = {
        'capture_dir': config.capture_dir,
        'img_size': config.img_size,
        'max_steps': config.max_steps,
        'use_enhanced_detection': config.use_enhanced_detection,
        'use_optical_flow': config.use_optical_flow,
        'use_path_planning': config.use_path_planning,
        'use_augmentation': config.use_augmentation,
        'save_debug_masks': False,  # Disable during training for speed
    }
    env = UnityCameraEnv(**env_kwargs)
    return env


# ========================= Training Functions ========================= #

def get_next_run_index(log_root: str, algo_prefix: str = "PPO") -> int:
    """Get next available run index"""
    os.makedirs(log_root, exist_ok=True)
    existing = [d for d in os.listdir(log_root) if d.startswith(algo_prefix)]
    indices = []
    for name in existing:
        parts = name.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            indices.append(int(parts[1]))
    return max(indices, default=0) + 1


def setup_directories(config: TrainingConfig, run_idx: int) -> Dict[str, Path]:
    """Setup directory structure for training run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{run_idx}_run_{timestamp}"

    dirs = {
        'tensorboard': Path(config.tensorboard_log) / f"PPO_{run_tag}",
        'models': Path("models") / run_tag,
        'logs': Path("logs") / run_tag,
        'checkpoints': Path("checkpoints") / run_tag,
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def train(config: TrainingConfig):
    """Main training function with enhanced features"""

    print("=" * 60)
    print("Enhanced Unity RL Training (Strict Passive)")
    print("=" * 60)

    # Setup run directories
    run_idx = get_next_run_index(config.tensorboard_log)
    dirs = setup_directories(config, run_idx)

    # Save configuration
    config_path = dirs['logs'] / 'config.json'
    config.save(str(config_path))
    print(f"Configuration saved to {config_path}")

    # Create training environment
    print("\nCreating training environment...")
    train_env = create_environment(config)

    # Add monitoring and time limit
    monitor_path = str(dirs['logs'] / 'monitor.csv')
    train_env = Monitor(train_env, filename=monitor_path)
    train_env = TimeLimit(train_env, max_episode_steps=config.max_steps or 500)
    train_env = DummyVecEnv([lambda: train_env])
    # Transpose (C,H,W) -> (H,W,C) for SB3 CNN
    train_env = VecTransposeImage(train_env)

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = create_environment(config)
    eval_env.eval_mode = True  # Set to evaluation mode
    eval_env = Monitor(eval_env)
    eval_env = TimeLimit(eval_env, max_episode_steps=config.max_steps or 500)
    eval_env = DummyVecEnv([lambda: eval_env])
    # Transpose (C,H,W) -> (H,W,C) for SB3 CNN
    eval_env = VecTransposeImage(eval_env)

    # Setup callbacks
    callbacks = []

    # Enhanced logging callback
    logging_callback = EnhancedLoggingCallback(
        log_dir=str(dirs['logs']),
        verbose=config.verbose
    )
    callbacks.append(logging_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(dirs['models']),
        log_path=str(dirs['logs']),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
        render=False,
        verbose=config.verbose
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=str(dirs['checkpoints']),
        name_prefix='ppo_checkpoint',
        verbose=config.verbose
    )
    callbacks.append(checkpoint_callback)

    # Curriculum learning callback
    if config.use_curriculum:
        curriculum_callback = AdaptiveCurriculumCallback(
            initial_max_steps=config.curriculum_initial_steps,
            target_max_steps=config.curriculum_target_steps,
            verbose=config.verbose
        )
        callbacks.append(curriculum_callback)
        print("Curriculum learning enabled")

    # Create PPO model with custom architecture
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])],
        activation_fn=nn.ReLU,
    )

    print("\nCreating PPO model...")
    model = PPO(
        "CnnPolicy",
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(dirs['tensorboard']),
        verbose=config.verbose
    )

    # Print model summary
    print("\nModel Architecture:")
    print(f"  Policy: CnnPolicy with custom network")
    print(f"  Observation space: {train_env.observation_space}")
    print(f"  Action space: {train_env.action_space}")
    try:
        total_params = sum(p.numel() for p in model.policy.parameters())
        print(f"  Total parameters: {total_params:,}")
    except Exception:
        pass

    # Training
    print("\n" + "=" * 60)
    print(f"Starting training for {config.total_timesteps:,} timesteps")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            tb_log_name=f"PPO_run_{run_idx}",
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_model_path = dirs['models'] / 'final_model.zip'
    model.save(str(final_model_path))
    print(f"\nFinal model saved to {final_model_path}")

    # Create summary report
    create_training_summary(dirs, config, run_idx)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results saved to: {dirs['logs']}")
    print(f"TensorBoard logs: {dirs['tensorboard']}")
    print("=" * 60)


def create_training_summary(dirs: Dict[str, Path], config: TrainingConfig, run_idx: int):
    """Create a summary report of the training run"""
    summary = {
        'run_index': run_idx,
        'timestamp': datetime.now().isoformat(),
        'config': config.to_dict(),
        'directories': {k: str(v) for k, v in dirs.items()},
        'enhanced_features': {
            'enhanced_detection': config.use_enhanced_detection,
            'optical_flow': config.use_optical_flow,
            'path_planning': config.use_path_planning,
            'augmentation': config.use_augmentation,
            'curriculum': config.use_curriculum,
        }
    }

    # Add training metrics if available
    monitor_file = dirs['logs'] / 'monitor.csv'
    if monitor_file.exists():
        import pandas as pd
        try:
            df = pd.read_csv(monitor_file, skiprows=1)  # Skip header comment
            if not df.empty:
                summary['training_metrics'] = {
                    'total_episodes': int(len(df)),
                    'mean_reward': float(df['r'].mean()),
                    'std_reward': float(df['r'].std()),
                    'max_reward': float(df['r'].max()),
                    'min_reward': float(df['r'].min()),
                    'mean_length': float(df['l'].mean()),
                }
        except Exception:
            pass

    # Save summary
    summary_path = dirs['logs'] / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nTraining summary saved to {summary_path}")


# ========================= Main Entry Point ========================= #

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Enhanced training script for Unity autonomous driving (Strict Passive)"
    )

    # Environment arguments
    parser.add_argument("--capture_dir", type=str, default="../../Assets/Captures",
                        help="Directory containing Unity image captures")
    parser.add_argument("--img_size", type=int, nargs=2, default=[84, 84],
                        help="Image size for observations (width height)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum steps per episode")

    # Enhanced features (defaults chosen for passive)
    parser.add_argument("--use_enhanced_detection", action="store_true", default=True,
                        help="Use enhanced lane detection with edge detection and Hough transform")
    parser.add_argument("--no_enhanced_detection", dest="use_enhanced_detection",
                        action="store_false",
                        help="Disable enhanced detection")
    parser.add_argument("--use_optical_flow", action="store_true", default=False,
                        help="Enable optical flow for temporal consistency")
    parser.add_argument("--no_optical_flow", dest="use_optical_flow",
                        action="store_false",
                        help="Disable optical flow")
    parser.add_argument("--use_path_planning", action="store_true", default=False,
                        help="Enable Bezier curve path planning (for reward only, not action blending)")
    parser.add_argument("--no_path_planning", dest="use_path_planning",
                        action="store_false",
                        help="Disable path planning")
    parser.add_argument("--use_augmentation", action="store_true", default=True,
                        help="Enable data augmentation during training")
    parser.add_argument("--no_augmentation", dest="use_augmentation",
                        action="store_false",
                        help="Disable data augmentation")

    # PPO hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2.5e-4,
                        help="Learning rate for PPO")
    parser.add_argument("--n_steps", type=int, default=512,
                        help="Number of steps to run for each environment per update")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Minibatch size for PPO")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of epochs for PPO")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda parameter")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clipping parameter")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="Value function coefficient")

    # Training settings
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps")
    parser.add_argument("--save_freq", type=int, default=10_000,
                        help="Save checkpoint every N timesteps")
    parser.add_argument("--eval_freq", type=int, default=5_000,
                        help="Evaluate every N timesteps")
    parser.add_argument("--eval_episodes", type=int, default=5,
                        help="Number of episodes for evaluation")

    # Curriculum learning
    parser.add_argument("--use_curriculum", action="store_true", default=False,
                        help="Enable curriculum learning")
    parser.add_argument("--curriculum_initial", type=int, default=100,
                        help="Initial max steps for curriculum learning")
    parser.add_argument("--curriculum_target", type=int, default=500,
                        help="Target max steps for curriculum learning")

    # Model loading (optional)
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to pre-trained model to continue training")
    parser.add_argument("--model_path", type=str, default="models/final_ppo_unity_model.zip",
                        help="(Deprecated) Manual model save path (auto-named by run)")

    # Logging
    parser.add_argument("--tensorboard_log", type=str, default="./tensorboard_logs",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level (0=none, 1=info, 2=debug)")

    # Debug options
    parser.add_argument("--save_debug_masks", action="store_true", default=False,
                        help="Save lane mask debug images during training")
    parser.add_argument("--test_env", action="store_true", default=False,
                        help="Test environment setup without training")

    args = parser.parse_args()

    # Test environment if requested
    if args.test_env:
        test_environment(args)
        return

    # Create configuration
    config = TrainingConfig(
        capture_dir=args.capture_dir,
        img_size=tuple(args.img_size),
        max_steps=args.max_steps,
        use_enhanced_detection=args.use_enhanced_detection,
        use_optical_flow=args.use_optical_flow,
        use_path_planning=args.use_path_planning,
        use_augmentation=args.use_augmentation,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        total_timesteps=args.timesteps,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        use_curriculum=args.use_curriculum,
        curriculum_initial_steps=args.curriculum_initial,
        curriculum_target_steps=args.curriculum_target,
        tensorboard_log=args.tensorboard_log,
        verbose=args.verbose,
    )

    # Load existing model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        config.loaded_from = args.load_model  # for provenance

    # Start training
    try:
        train(config)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_environment(args):
    """Test environment setup and features"""
    print("=" * 60)
    print("Testing Environment Setup (Strict Passive)")
    print("=" * 60)

    # Create configuration
    config = TrainingConfig(
        capture_dir=args.capture_dir,
        img_size=tuple(args.img_size),
        use_enhanced_detection=args.use_enhanced_detection,
        use_optical_flow=args.use_optical_flow,
        use_path_planning=args.use_path_planning,
        use_augmentation=False,   # Disable for testing speed
    )

    # Create environment
    print("\nCreating test environment...")
    env = create_environment(config)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run test episodes
    print("\nRunning test episodes...")
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        obs, info = env.reset()
        print(f"  Initial info keys: {list(info.keys())}")

        total_reward = 0.0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"  Step {step + 1}: Reward={reward:.3f}, "
                  f"Coverage={info.get('coverage', 0)}, "
                  f"Confidence={info.get('confidence', 0):.2f}")
            if terminated or truncated:
                break
        print(f"  Episode reward: {total_reward:.2f}")

    env.close()
    print("\nEnvironment test complete!")

    print("\nEnhanced Features Status:")
    print(f"  Enhanced Detection: {'✓' if config.use_enhanced_detection else '✗'}")
    print(f"  Optical Flow: {'✓' if config.use_optical_flow else '✗'}")
    print(f"  Path Planning: {'✓' if config.use_path_planning else '✗'}")
    print(f"  Data Augmentation: {'✓' if config.use_augmentation else '✗'}")

    print("\nDebug masks saved to: Logs/DebugMasks/")


if __name__ == "__main__":
    main()
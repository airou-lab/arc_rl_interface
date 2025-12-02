# Save as test_train_vec.py
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from unity_dense_env import UnityDenseEnv

def make_env():
    def _init():
        return UnityDenseEnv(
            host='127.0.0.1',
            port=5556,
            img_width=128,
            img_height=128,
            max_steps=500,
            verbose=False
        )
    return _init

# Create vectorized environment (this is what train_policy_RNN.py does)
vec_env = DummyVecEnv([make_env()])

print("Creating model...")
model = RecurrentPPO(
    "MultiInputLstmPolicy",
    vec_env,
    n_steps=64,
    batch_size=64,
    verbose=1
)

print("Starting training...")
model.learn(total_timesteps=1000)
print("Training complete!")
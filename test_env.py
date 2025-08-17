from unity_camera_env import UnityCameraEnv
import os
import csv

def main():
    env = UnityCameraEnv(capture_dir="/Users/aaron/Documents/arc_rl_unity/Assets/Captures")
    obs, info = env.reset()

    # Write intrinsics to a separate CSV
    intrinsics_path = os.path.join(env.capture_dir, "camera_intrinsics_log.csv")
    with open(intrinsics_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=info.keys())
        writer.writeheader()
        writer.writerow(info)
    print("[INFO] Intrinsics written to:", intrinsics_path)

    # Prepare pose log
    pose_path = os.path.join(env.capture_dir, "camera_pose_log.csv")
    with open(pose_path, mode='w', newline='') as file:
        pose_writer = csv.writer(file)
        pose_writer.writerow(["frame_id", "x", "y", "z", "qx", "qy", "qz", "qw"])

        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated

            # Log pose (dummy values if not available)
            frame_id = f"CameraRGB_{info['step_idx']:04d}"
            pose = info.get("pose", (0, 0, 0, 0, 0, 0, 1))
            pose_writer.writerow([frame_id] + list(pose))

            print(f"Step {info['step_idx']}: Action={action}, Reward={reward:.2f}, FOV={info['fov']} fx={info['fx']:.2f}")

    print("Total Reward:", total_reward)
    print("[INFO] Pose log written to:", pose_path)
    env.close()

if __name__ == "__main__":
    main()
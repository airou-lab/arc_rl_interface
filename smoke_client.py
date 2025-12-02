"""
Minimal client to sanity-check the Unity TCP interface.

Protocol (Unity -> Python):
- JPEG length: 4 bytes, big-endian uint32
- JPEG data: variable
- Telemetry: 12 floats, big-endian (48 bytes)
- Reward: 1 float, big-endian (4 bytes)
- Done: 1 byte
- Truncated: 1 byte

Protocol (Python -> Unity):
- Reset: 'R' (1 byte)
- Action: 'A' + steer + thr + brk (1 + 12 bytes, little-endian floats)
"""
from __future__ import annotations
import socket
import struct
import time

HOST, PORT = "127.0.0.1", 5556

def read_exact(s: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from socket"""
    b = bytearray()
    while len(b) < n:
        chunk = s.recv(n - len(b))
        if not chunk:
            raise ConnectionError("socket closed")
        b.extend(chunk)
    return bytes(b)

def recv_observation(s: socket.socket):
    """Receive full observation from Unity"""
    # Read JPEG length (4 bytes, big-endian)
    jpeg_len = struct.unpack(">I", read_exact(s, 4))[0]

    # Read JPEG data
    if jpeg_len > 0:
        _ = read_exact(s, jpeg_len)

    # Read telemetry (12 floats, big-endian = 48 bytes)
    tel_bytes = read_exact(s, 48)
    telemetry = struct.unpack(">12f", tel_bytes)

    # Read reward (4 bytes, big-endian)
    reward = struct.unpack(">f", read_exact(s, 4))[0]

    # Read done and truncated (1 byte each)
    done = bool(read_exact(s, 1)[0])
    truncated = bool(read_exact(s, 1)[0])

    return telemetry, reward, done, truncated

def send_action(s: socket.socket, steer: float, throttle: float, brake: float = 0.0):
    """Send action to Unity: 'A' + 3 floats (little-endian)"""
    msg = b'A' + struct.pack('<fff', float(steer), float(throttle), float(brake))
    s.sendall(msg)

def main():
    print(f"Connecting to Unity at {HOST}:{PORT}...")

    with socket.create_connection((HOST, PORT), timeout=30) as s:
        print("Connected to Unity.")

        # Send reset command
        print("Sending reset...")
        s.sendall(b"R")

        # Read first observation
        telemetry, reward, done, truncated = recv_observation(s)

        speed = telemetry[3]
        goal_cos = telemetry[0]
        goal_dist = telemetry[2]

        print(f"First frame:")
        print(f"  Speed: {speed:.2f} m/s")
        print(f"  Goal cos: {goal_cos:.3f}, dist: {goal_dist:.1f}m")
        print(f"  Reward: {reward:.4f}")
        print(f"  Done: {done}, Truncated: {truncated}")

        if done or truncated:
            print("\n⚠️  Episode ended immediately!")
            print("Check Unity: goal might be too close or car spawns in collision.")
            return

        print("\nRunning test drive (alternating steering)...")

        for t in range(400):
            # Alternate steering left/right every 60 frames
            steer = 0.3 if (t // 60) % 2 == 0 else -0.3
            throttle = 0.4
            brake = 0.0

            send_action(s, steer, throttle, brake)

            telemetry, reward, done, truncated = recv_observation(s)

            speed = telemetry[3]
            goal_dist = telemetry[2]
            lat_err = telemetry[8]

            if t % 20 == 0:
                print(f"t={t:3d} | speed={speed:5.2f} | goal={goal_dist:6.1f}m | lat_err={lat_err:+5.2f} | r={reward:+.3f}")

            if done or truncated:
                status = "GOAL REACHED" if done and reward > 5 else "TERMINATED"
                print(f"\n{status} at t={t}")
                break

            time.sleep(0.02)  # ~50Hz

        print("\nTest complete!")

if __name__ == "__main__":
    main()
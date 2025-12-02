#!/usr/bin/env python3
"""
Debug script to test waypoint visualization in Unity
Run this while Unity is in play mode to test if waypoints are being displayed
CORRECTED VERSION: Properly handles the Unity protocol
"""

import socket
import struct
import numpy as np
import time


def send_waypoints_debug(host="127.0.0.1", port=5556):
    """Send test waypoints to Unity to verify visualization is working"""

    print(f"Connecting to Unity at {host}:{port}...")

    # Connect to Unity
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    print("Connected! Sending reset...")

    # Send reset
    sock.sendall(b'R')

    # Wait a bit for Unity to process reset
    time.sleep(0.5)

    print("Sending waypoints and actions in a loop...")
    print("You should see green waypoints moving in Unity scene view...")
    print("Press Ctrl+C to stop")

    try:
        for iteration in range(100):
            # Generate test waypoints (5 waypoints moving in a pattern)
            t = iteration * 0.1
            waypoints = []
            for i in range(5):
                # Create a path that curves slightly
                x = np.sin(t + i * 0.3) * 1.5  # Lateral movement
                y = 2.0 + i * 2.0  # Forward movement (2m spacing)
                waypoints.extend([x, y])

            # Send waypoint message
            msg = b'W'  # Waypoint command
            msg += struct.pack('<B', 5)  # 5 waypoints (little-endian byte)

            # Add waypoints (little-endian floats)
            for val in waypoints:
                msg += struct.pack('<f', float(val))

            sock.sendall(msg)
            print(f"Iteration {iteration}: Sent waypoints: [{waypoints[0]:.2f}, {waypoints[1]:.2f}] ...")

            # Send a dummy action to keep the episode running
            action_msg = b'A'
            # Small steering, moderate throttle, no brake
            steer = np.sin(t * 0.5) * 0.3  # Gentle steering
            throttle = 0.4  # Moderate speed
            brake = 0.0

            action_msg += struct.pack('<fff', steer, throttle, brake)
            sock.sendall(action_msg)

            # Wait for Unity's response (we'll read and discard it)
            try:
                # Read the response header (4 bytes for JPEG length)
                jpeg_len_bytes = sock.recv(4)
                if len(jpeg_len_bytes) < 4:
                    print("Unity closed connection")
                    break

                jpeg_len = struct.unpack('>I', jpeg_len_bytes)[0]

                # Read and discard the full response
                # JPEG data
                if jpeg_len > 0:
                    jpeg_data = sock.recv(jpeg_len)

                # Telemetry: 12 floats * 4 bytes = 48 bytes
                telemetry = sock.recv(48)

                # Reward (4 bytes), done (1 byte), truncated (1 byte)
                rest = sock.recv(6)

                # Check if episode ended
                if len(rest) >= 6:
                    done = rest[4]
                    truncated = rest[5]
                    if done or truncated:
                        print(f"Episode ended (done={done}, truncated={truncated}), sending reset...")
                        sock.sendall(b'R')
                        time.sleep(0.5)

            except socket.timeout:
                print("Timeout reading response, continuing...")
            except Exception as e:
                print(f"Error reading response: {e}, continuing...")

            # Control update rate
            time.sleep(0.05)  # 20 Hz update rate

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()
        print("Connection closed")


if __name__ == "__main__":
    send_waypoints_debug()
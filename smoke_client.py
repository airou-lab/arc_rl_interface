"""
Minimal client to sanity-check the Unity TCP interface.

- Sends a reset 'R'
- Then alternates steering left/right while holding throttle
- Prints reward, done, truncated every ~0.03 s
"""
from __future__ import annotations
import socket
import struct
import time

HOST, PORT = "127.0.0.1", 5556

def be_f32(x: float) -> bytes:
    return struct.pack(">f", float(x))

def read_exact(s: socket.socket, n: int) -> bytes:
    b = bytearray()
    while len(b) < n:
        chunk = s.recv(n - len(b))
        if not chunk:
            raise ConnectionError("socket closed")
        b.extend(chunk)
    return bytes(b)

def main():
    with socket.create_connection((HOST, PORT), timeout=10) as s:
        print("Connected to Unity.")
        s.sendall(b"R")  # reset

        # read first obs
        n = struct.unpack(">I", read_exact(s, 4))[0]
        if n:
            _ = read_exact(s, n)
        tail = read_exact(s, 6)
        rew = struct.unpack(">f", tail[:4])[0]
        done, trunc = tail[4], tail[5]
        print(f"First frame: reward={rew}, done={bool(done)}, truncated={bool(trunc)}")

        for t in range(400):
            steer = 0.6 if (t // 60) % 2 == 0 else -0.6
            throttle = 0.5
            s.sendall(be_f32(steer) + be_f32(throttle))

            n = struct.unpack(">I", read_exact(s, 4))[0]
            if n: _ = read_exact(s, n)
            tail = read_exact(s, 6)
            r = struct.unpack(">f", tail[:4])[0]
            d, tr = bool(tail[4]), bool(tail[5])
            print(f"t={t} reward={r:+.3f} done={d} trunc={tr}")
            if d or tr:
                break
            time.sleep(0.03)

if __name__ == "__main__":
    main()
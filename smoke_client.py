import socket, struct, cv2, numpy as np

HOST, PORT = "127.0.0.1", 5556

def recv_exact(s, n):
    data = b""
    while len(data) < n:
        chunk = s.recv(n - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data += chunk
    return data

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("Connected to Unity.")
    # Reset -> Unity should send first frame + tail
    s.sendall(b"R")
    L = struct.unpack("!I", recv_exact(s, 4))[0]
    jpeg = recv_exact(s, L)
    reward = struct.unpack("!f", recv_exact(s, 4))[0]
    done = bool(recv_exact(s,1)[0]); truncated = bool(recv_exact(s,1)[0])
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    print(f"First frame: {None if img is None else img.shape}, reward={reward}, done={done}, truncated={truncated}")

    # Drive forward gently for 100 steps
    for t in range(100):
        steer, throttle = 0.0, 0.3
        s.sendall(struct.pack("!ff", float(steer), float(throttle)))
        L = struct.unpack("!I", recv_exact(s, 4))[0]
        jpeg = recv_exact(s, L)
        tail = recv_exact(s, 6)
        reward = struct.unpack("!f", tail[:4])[0]
        done = bool(tail[4]); truncated = bool(tail[5])
        if t % 10 == 0:
            print(f"t={t} reward={reward:.3f} done={done} trunc={truncated}")
        if done or truncated:
            break
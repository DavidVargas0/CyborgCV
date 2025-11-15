#!/usr/bin/env python3

import socket

# =======================
# CONFIGURATION
# =======================

# Replace this with your Raspberry Pi's Bluetooth MAC address.
# Example: "B8:27:EB:12:34:56"
PI_BLUETOOTH_ADDR = "EE:24:62:F7:4B:91"  # <-- PUT YOUR PI'S ADDRESS HERE
RFCOMM_CHANNEL = 1  # must match the server


def send_color(sock, r, g, b):
    """Send a single [R, G, B] packet as 3 raw bytes."""
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    sock.send(bytes([r, g, b]))
    print(f"Sent color R={r}, G={g}, B={b}")


def main():
    print(f"Connecting to {PI_BLUETOOTH_ADDR} on RFCOMM channel {RFCOMM_CHANNEL}...")

    # On Windows, use AF_BTH (Bluetooth) + BTPROTO_RFCOMM
    sock = socket.socket(
        socket.AF_BTH,
        socket.SOCK_STREAM,
        socket.BTPROTO_RFCOMM,
    )
    sock.connect((PI_BLUETOOTH_ADDR, RFCOMM_CHANNEL))
    print("Connected!\n")

    try:
        print("Enter colors as: R G B  (0-255). Type 'quit' to exit.")
        while True:
            line = input("> ").strip()
            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                break

            parts = line.split()
            if len(parts) != 3:
                print("Please enter exactly 3 numbers: R G B")
                continue

            try:
                r, g, b = map(int, parts)
            except ValueError:
                print("Each value must be an integer between 0 and 255.")
                continue

            send_color(sock, r, g, b)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        sock.close()
        print("Disconnected.")


if __name__ == "__main__":
    main()

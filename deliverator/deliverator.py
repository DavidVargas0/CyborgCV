#!/usr/bin/env python3
import socket
import sys

# ===== CONFIG =====
PI_IP = "brianpi"  # <-- CHANGE THIS to your Pi's IP address
UDP_PORT = 5005         # Must match the Pi's port

def send_color(r, g, b):
    # Clamp to 0-255 just in case
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    payload = bytes([r, g, b])  # Super compact 3-byte payload
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(payload, (PI_IP, UDP_PORT))
    sock.close()
    print(f"Sent color R={r} G={g} B={b} to {PI_IP}:{UDP_PORT}")

def parse_hex_color(hex_str):
    """Accepts 'RRGGBB' or '#RRGGBB'."""
    hex_str = hex_str.strip().lstrip('#')
    if len(hex_str) != 6:
        raise ValueError("Hex color must be 6 characters like 'FF0000' or '#FF0000'")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return r, g, b

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Usage: python send_color.py FF0000
        r, g, b = parse_hex_color(sys.argv[1])
        send_color(r, g, b)
    elif len(sys.argv) == 4:
        # Usage: python send_color.py 255 0 0
        r = int(sys.argv[1])
        g = int(sys.argv[2])
        b = int(sys.argv[3])
        send_color(r, g, b)
    else:
        print("Usage:")
        print("  python send_color.py RRGGBB")
        print("  python send_color.py R G B")
        print("Examples:")
        print("  python send_color.py FF0000      # red")
        print("  python send_color.py 0 255 0     # green")

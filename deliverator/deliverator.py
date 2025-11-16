#!/usr/bin/env python3
import socket
import sys

# ===== CONFIG =====
PI_IP = "brianpi"  # <-- change to your Pi's IP
UDP_PORT = 5005

def send_color_indices(r, g, b, indices):
    # Clamp color
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    # Clamp indices and remove duplicates for safety/compactness
    clean_indices = []
    for idx in indices:
        if 0 <= idx <= 255:  # fits in one byte
            clean_indices.append(idx)

    n = len(clean_indices)
    if n > 255:
        clean_indices = clean_indices[:255]
        n = 255

    # Build payload: R,G,B,N,indices...
    payload = bytes([r, g, b, n] + clean_indices)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(payload, (PI_IP, UDP_PORT))
    sock.close()
    print(f"Sent color ({r},{g},{b}) to indices {clean_indices} -> {PI_IP}:{UDP_PORT}")

def parse_hex_color(hex_str):
    hex_str = hex_str.strip().lstrip('#')
    if len(hex_str) != 6:
        raise ValueError("Hex color must be 6 characters like 'FF0000' or '#FF0000'")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return r, g, b

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python send_color_indices.py RRGGBB idx idx idx ...")
        print("  python send_color_indices.py R G B idx idx idx ...")
        print("Examples:")
        print("  python send_color_indices.py FF0000 0 1 2 3")
        print("  python send_color_indices.py 0 255 0 8 9 10 11")
        sys.exit(1)

    args = sys.argv[1:]

    # Hex mode: first arg has length 6 or starts with '#'
    if len(args[0].lstrip('#')) == 6 and all(c in "0123456789abcdefABCDEF#" for c in args[0]):
        r, g, b = parse_hex_color(args[0])
        idx_args = args[1:]
    else:
        if len(args) < 4:
            print("Not enough arguments for R G B mode.")
            sys.exit(1)
        r = int(args[0])
        g = int(args[1])
        b = int(args[2])
        idx_args = args[3:]

    indices = [int(x) for x in idx_args]
    send_color_indices(r, g, b, indices)

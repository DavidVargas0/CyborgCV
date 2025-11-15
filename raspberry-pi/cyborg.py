#!/usr/bin/env python3
import socket
import board
import neopixel

# ===== NeoPixel config =====
LED_PIN = board.D18      # GPIO18 (pin 12) – change if needed
LED_COUNT = 64           # Set to 16 or 64 depending on your setup
BRIGHTNESS = 0.2         # 0.0 to 1.0
AUTO_WRITE = False       # We'll call show() manually
PIXEL_ORDER = neopixel.GRB  # Typical for many NeoPixels

pixels = neopixel.NeoPixel(
    LED_PIN,
    LED_COUNT,
    brightness=BRIGHTNESS,
    auto_write=AUTO_WRITE,
    pixel_order=PIXEL_ORDER
)

# Start with LEDs off
pixels.fill((0, 0, 0))
pixels.show()

# ===== UDP config =====
UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 5005     # Arbitrary, but must match sender

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for UDP color packets on port {UDP_PORT}...")

try:
    while True:
        data, addr = sock.recvfrom(1024)  # Buffer size is more than enough
        if len(data) < 3:
            # Invalid / too short, ignore
            continue

        # First 3 bytes are R,G,B (0–255)
        r, g, b = data[0], data[1], data[2]

        # Set ALL LEDs to this color
        pixels.fill((r, g, b))
        pixels.show()

        print(f"Set color to R={r} G={g} B={b} from {addr}")

except KeyboardInterrupt:
    print("Exiting, turning LEDs off...")
    pixels.fill((0, 0, 0))
    pixels.show()

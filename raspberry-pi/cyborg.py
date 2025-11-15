#!/usr/bin/env python3

import time
import board
import neopixel

# =======================
# CONFIGURATION
# =======================

# Number of LEDs in your matrix (8x8 = 64)
NUM_PIXELS = 64

# GPIO pin used for the NeoPixel data line.
# On Raspberry Pi, board.D18 corresponds to physical pin 12 (GPIO18).
PIXEL_PIN = board.D18

# NeoPixels are usually powered by 5V
PIXEL_ORDER = neopixel.GRB  # Most Adafruit NeoPixels use GRB

# =======================
# SETUP
# =======================

pixels = neopixel.NeoPixel(
    PIXEL_PIN,
    NUM_PIXELS,
    brightness=0.3,   # 0.0 to 1.0
    auto_write=False, # We'll call show() manually
    pixel_order=PIXEL_ORDER
)

# =======================
# MAIN
# =======================

try:
    # Set every pixel to blue (R, G, B)
    blue_color = (0, 0, 255)

    for i in range(NUM_PIXELS):
        pixels[i] = blue_color

    pixels.show()
    print("All pixels set to blue. Press Ctrl+C to turn them off and exit.")

    # Keep running so LEDs stay lit until you stop the program
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    # Turn off all pixels on exit
    pixels.fill((0, 0, 0))
    pixels.show()
    print("\nPixels cleared. Exiting.")

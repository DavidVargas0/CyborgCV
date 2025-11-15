import time
import board
import neopixel

MATRIX_PIN = board.D6
NUM_PIXELS = 8 * 8

pixels = neopixel.NeoPixel(MATRIX_PIN, NUM_PIXELS, brightness=0.2, auto_write=True)

# Fill the whole matrix blue
pixels.fill((0, 0, 40))

while True:
    time.sleep(1)

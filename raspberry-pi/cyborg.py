#!/usr/bin/env python3

import time
import socket

import board
import neopixel

# =======================
# LED CONFIGURATION
# =======================

NUM_PIXELS = 64               # 8x8 matrix
PIXEL_PIN = board.D18         # GPIO18 (physical pin 12)
PIXEL_ORDER = neopixel.GRB    # Most Adafruit NeoPixels use GRB

pixels = neopixel.NeoPixel(
    PIXEL_PIN,
    NUM_PIXELS,
    brightness=0.3,
    auto_write=False,
    pixel_order=PIXEL_ORDER
)

def set_color(r, g, b):
    """Set all pixels to the given RGB color."""
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    pixels.fill((r, g, b))
    pixels.show()
    print(f"Set color to R={r}, G={g}, B={b}")


# =======================
# BLUETOOTH SERVER (no PyBluez)
# =======================

RFCOMM_CHANNEL = 1  # same as before


def run_server():
    # Native Bluetooth RFCOMM socket (Linux / BlueZ)
    server_sock = socket.socket(
        socket.AF_BLUETOOTH,
        socket.SOCK_STREAM,
        socket.BTPROTO_RFCOMM,
    )

    # Use BDADDR_ANY so we bind on any local Bluetooth adapter
    bdaddr_any = getattr(socket, "BDADDR_ANY", "00:00:00:00:00:00")
    server_sock.bind((bdaddr_any, RFCOMM_CHANNEL))
    server_sock.listen(1)

    print("Bluetooth NeoPixel server started.")
    print(f"Waiting for a connection on RFCOMM channel {RFCOMM_CHANNEL}...")

    client_sock = None
    try:
        client_sock, client_info = server_sock.accept()
        print(f"Accepted connection from {client_info}")

        while True:
            data = client_sock.recv(3)
            if not data:
                print("Client disconnected.")
                break

            while len(data) < 3:
                more = client_sock.recv(3 - len(data))
                if not more:
                    break
                data += more

            if len(data) != 3:
                print("Incomplete packet received, closing connection.")
                break

            r, g, b = data[0], data[1], data[2]
            print(f"Received packet: R={r}, G={g}, B={b}")
            set_color(r, g, b)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: shutting down server.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pixels.fill((0, 0, 0))
        pixels.show()
        if client_sock is not None:
            client_sock.close()
        server_sock.close()
        print("Server stopped. LEDs cleared.")


if __name__ == "__main__":
    run_server()

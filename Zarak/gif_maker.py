#!/usr/bin/env python3
"""
Script to create an animated GIF from all PNG files in a folder
"""

import os
from pathlib import Path
from PIL import Image
import argparse


def create_gif_from_pngs(input_folder, output_file="output.gif", duration=500, loop=0):
    """
    Create an animated GIF from all PNG files in a folder

    Args:
        input_folder: Path to folder containing PNG files
        output_file: Output GIF filename (default: output.gif)
        duration: Duration of each frame in milliseconds (default: 500)
        loop: Number of times to loop (0 = infinite, default: 0)
    """
    # Get all PNG files in the folder
    input_path = Path(input_folder)
    png_files = sorted(input_path.glob("*.png"))

    if not png_files:
        print(f"No PNG files found in {input_folder}")
        return

    print(f"Found {len(png_files)} PNG files:")
    for png in png_files:
        print(f"  - {png.name}")

    # Load all images
    images = []
    for png_file in png_files:
        try:
            img = Image.open(png_file)
            # Convert to RGB if necessary (GIF doesn't support RGBA well)
            if img.mode == 'RGBA':
                # Create a white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                images.append(background)
            else:
                images.append(img.convert('RGB'))
        except Exception as e:
            print(f"Error loading {png_file}: {e}")

    if not images:
        print("No images could be loaded successfully")
        return

    # Save as animated GIF
    print(f"\nCreating GIF: {output_file}")
    print(f"Frame duration: {duration}ms")
    print(f"Loop: {'infinite' if loop == 0 else loop}")

    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )

    print(f"\n✓ GIF created successfully: {output_file}")
    print(f"  Size: {os.path.getsize(output_file) / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Create an animated GIF from PNG files in a folder"
    )
    parser.add_argument(
        "input_folder",
        help="Path to folder containing PNG files"
    )
    parser.add_argument(
        "-o", "--output",
        default="output.gif",
        help="Output GIF filename (default: output.gif)"
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=500,
        help="Duration of each frame in milliseconds (default: 500)"
    )
    parser.add_argument(
        "-l", "--loop",
        type=int,
        default=0,
        help="Number of times to loop (0 = infinite, default: 0)"
    )

    args = parser.parse_args()

    create_gif_from_pngs(
        args.input_folder,
        args.output,
        args.duration,
        args.loop
    )


if __name__ == "__main__":
    create_gif_from_pngs(
        input_folder=r"C:\Users\ZKasi\OneDrive - Thornton Tomasetti, Inc\Desktop\AEC tech Hackathon\Hackathon\annotated_frames",
        output_file="my_animation.gif",
        duration=1,  # milliseconds per frame
        loop=0  # 0 = infinite loop
    )
    # Remove this line: main()
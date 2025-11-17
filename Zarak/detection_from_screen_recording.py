import cv2
import numpy as np
import json
import os
import sys
from datetime import datetime


class ScreenDrawingTracker:
    def __init__(self, video_path):
        self.video_path = video_path

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {os.path.basename(video_path)}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps}, Frames: {self.total_frames}")
        print(f"Duration: {self.total_frames / self.fps:.2f}s")

        # Canvas detection
        self.canvas_bounds = None
        self.prev_frame = None

        # Drawing storage
        self.drawing_path = []
        self.all_drawn_pixels = set()

    def detect_white_canvas(self, frame):
        """Detect the white drawing canvas area"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest white area (canvas)
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50000:
                x, y, w, h = cv2.boundingRect(largest)

                if self.canvas_bounds is None:
                    self.canvas_bounds = (x, y, w, h)
                else:
                    # Smooth bounds
                    ox, oy, ow, oh = self.canvas_bounds
                    alpha = 0.1
                    self.canvas_bounds = (
                        int(ox * (1 - alpha) + x * alpha),
                        int(oy * (1 - alpha) + y * alpha),
                        int(ow * (1 - alpha) + w * alpha),
                        int(oh * (1 - alpha) + h * alpha)
                    )
                return True
        return False

    def detect_new_drawing(self, frame):
        """Detect new pixels drawn on canvas by comparing frames"""
        if self.prev_frame is None or self.canvas_bounds is None:
            self.prev_frame = frame.copy()
            return []

        x, y, w, h = self.canvas_bounds

        # Extract canvas regions
        curr_canvas = frame[y:y + h, x:x + w]
        prev_canvas = self.prev_frame[y:y + h, x:x + w]

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr_canvas, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_canvas, cv2.COLOR_BGR2GRAY)

        # Find difference (new drawing)
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

        # Find new drawn pixels
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Get coordinates of new pixels
        new_pixels = []
        points = np.where(thresh > 0)

        for py, px in zip(points[0], points[1]):
            # Check if it's a dark pixel (drawing, not erasing)
            if curr_gray[py, px] < 200:
                pixel_key = (px, py)
                if pixel_key not in self.all_drawn_pixels:
                    self.all_drawn_pixels.add(pixel_key)
                    new_pixels.append((px, py))

        self.prev_frame = frame.copy()
        return new_pixels

    def process_video(self):
        """Process video and extract drawing"""
        print("\n" + "=" * 60)
        print("Processing video...")
        print("=" * 60)

        frame_num = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = frame_num / self.fps

            # Detect canvas
            self.detect_white_canvas(frame)

            # Detect new drawing
            new_pixels = self.detect_new_drawing(frame)

            # Store coordinates
            for px, py in new_pixels:
                if self.canvas_bounds:
                    x, y, w, h = self.canvas_bounds
                    coord = {
                        'pixel': (int(px), int(py)),
                        'normalized': (float(px / w), float(py / h)),
                        'absolute': (int(x + px), int(y + py)),
                        'frame': int(frame_num),
                        'timestamp': float(timestamp)
                    }
                    self.drawing_path.append(coord)

            # Progress
            frame_num += 1
            progress = (frame_num / self.total_frames) * 100
            bar_len = 40
            filled = int(bar_len * frame_num / self.total_frames)
            bar = '█' * filled + '-' * (bar_len - filled)
            print(f"\r|{bar}| {progress:.1f}%", end='', flush=True)

        print("\n\n" + "=" * 60)
        print(f"Processing complete!")
        print(f"Total drawing points: {len(self.drawing_path)}")
        print("=" * 60)

        self.cap.release()

    def save_coordinates(self, output_file=None):
        """Save coordinates to JSON"""
        if not self.drawing_path:
            print("No drawing data to save!")
            return None

        if output_file is None:
            base = os.path.splitext(os.path.basename(self.video_path))[0]
            output_file = f"{base}_drawing.json"

        data = {
            'metadata': {
                'source_video': os.path.basename(self.video_path),
                'total_points': len(self.drawing_path),
                'video_fps': self.fps,
                'canvas_bounds': self.canvas_bounds,
                'timestamp': datetime.now().isoformat()
            },
            'coordinates': self.drawing_path
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Coordinates: {output_file}")
        return output_file

    def create_drawing_video(self, output_file=None):
        """Create clean video of just the drawing"""
        if not self.drawing_path or self.canvas_bounds is None:
            print("No drawing to visualize!")
            return None

        if output_file is None:
            base = os.path.splitext(os.path.basename(self.video_path))[0]
            output_file = f"{base}_drawing.mp4"

        print("\n" + "=" * 60)
        print("Creating drawing video...")
        print("=" * 60)

        x, y, w, h = self.canvas_bounds

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, self.fps, (w, h))

        # Create frames
        max_frame = max(p['frame'] for p in self.drawing_path)

        for frame_num in range(max_frame + 1):
            # White canvas
            canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

            # Draw all points up to current frame
            points_so_far = [p for p in self.drawing_path if p['frame'] <= frame_num]

            for point in points_so_far:
                px, py = point['pixel']
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(canvas, (px, py), 1, (0, 0, 0), -1)

            out.write(canvas)

            # Progress
            progress = ((frame_num + 1) / (max_frame + 1)) * 100
            bar_len = 40
            filled = int(bar_len * (frame_num + 1) / (max_frame + 1))
            bar = '█' * filled + '-' * (bar_len - filled)
            print(f"\r|{bar}| {progress:.1f}%", end='', flush=True)

        out.release()
        print(f"\n✓ Drawing video: {output_file}")
        return output_file


def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python screen_drawing_tracker.py <video_file>")
    #     print("\nExample: python screen_drawing_tracker.py recording.mp4")
    #     return

    video_path = r"C:\Users\ZKasi\OneDrive - Thornton Tomasetti, Inc\Desktop\AEC tech Hackathon\Recording 2025-11-15 132630.mp4"

    print("\n" + "=" * 60)
    print("SCREEN DRAWING TRACKER")
    print("=" * 60)
    print("Extracts drawing from Paint/canvas screen recordings")
    print("=" * 60)

    try:
        tracker = ScreenDrawingTracker(video_path)
        tracker.process_video()

        json_file = tracker.save_coordinates()
        video_file = tracker.create_drawing_video()

        print("\n" + "=" * 60)
        print("DONE! 🎉")
        print("=" * 60)
        print(f"\nFiles created:")
        if json_file:
            print(f"  📄 {json_file}")
        if video_file:
            print(f"  🎥 {video_file}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
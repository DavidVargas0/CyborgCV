import cv2
import numpy as np
import json
from datetime import datetime
import time


class PenTracker:
    def __init__(self):
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Paper boundary variables
        self.paper_contour = None
        self.paper_bounds = None

        # Pen tracking variables (default: blue pen)
        self.pen_lower = np.array([100, 150, 50])
        self.pen_upper = np.array([130, 255, 255])

        # Drawing path storage
        self.drawing_path = []
        self.current_pen_pos = None
        self.pen_in_bounds = False

        # Display settings
        self.show_paper_detection = True
        self.show_pen_tracking = True

        # FPS tracking
        self.prev_time = time.time()
        self.fps = 0

    def detect_white_paper(self, frame):
        """Detect the white paper boundaries in the frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold to detect white areas
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (assumed to be the paper)
            largest_contour = max(contours, key=cv2.contourArea)

            # Only consider if the contour is large enough
            if cv2.contourArea(largest_contour) > 10000:
                # Approximate the contour to a polygon
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)

                self.paper_contour = largest_contour
                self.paper_bounds = (x, y, w, h)

                return True

        return False

    def detect_pen(self, frame):
        """Detect the pen position using color tracking"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for pen color
        mask = cv2.inRange(hsv, self.pen_lower, self.pen_upper)

        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours of the pen
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (pen tip)
            largest_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_contour) > 100:
                # Get the center of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    self.current_pen_pos = (cx, cy)
                    return True

        self.current_pen_pos = None
        return False

    def is_pen_in_paper_bounds(self):
        """Check if the pen is within the paper boundaries"""
        if self.current_pen_pos is None or self.paper_bounds is None:
            return False

        x, y, w, h = self.paper_bounds
        px, py = self.current_pen_pos

        # Check if pen position is within paper bounds
        if x <= px <= x + w and y <= py <= y + h:
            return True
        return False

    def get_relative_coordinates(self):
        """Get pen coordinates relative to paper boundaries"""
        if self.current_pen_pos is None or self.paper_bounds is None:
            return None

        x, y, w, h = self.paper_bounds
        px, py = self.current_pen_pos

        # Calculate relative position within the paper
        rel_x = px - x
        rel_y = py - y

        # Also calculate normalized coordinates (0-1 range)
        norm_x = rel_x / w if w > 0 else 0
        norm_y = rel_y / h if h > 0 else 0

        return {
            'pixel': (rel_x, rel_y),
            'normalized': (norm_x, norm_y),
            'absolute': (px, py),
            'timestamp': time.time()
        }

    def record_coordinate(self):
        """Record the current pen coordinate if in bounds"""
        if self.pen_in_bounds:
            coords = self.get_relative_coordinates()
            if coords:
                self.drawing_path.append(coords)

    def draw_visualizations(self, frame):
        """Draw all visualizations on the frame"""
        display_frame = frame.copy()

        # Draw paper boundary
        if self.paper_bounds is not None and self.show_paper_detection:
            x, y, w, h = self.paper_bounds
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(display_frame, "Paper Detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw pen position and status
        if self.current_pen_pos is not None:
            px, py = self.current_pen_pos

            # Choose color based on bounds status
            if self.pen_in_bounds:
                color = (0, 255, 0)  # Green when in bounds
                status_text = "IN BOUNDS"
            else:
                color = (0, 0, 255)  # Red when out of bounds
                status_text = "OUT OF BOUNDS"

            # Draw pen position indicator
            cv2.circle(display_frame, (px, py), 10, color, -1)
            cv2.circle(display_frame, (px, py), 15, color, 2)

            # Draw status text
            cv2.putText(display_frame, status_text, (px + 20, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw coordinates
            if self.pen_in_bounds:
                coords = self.get_relative_coordinates()
                if coords:
                    coord_text = f"Rel: ({coords['pixel'][0]}, {coords['pixel'][1]})"
                    cv2.putText(display_frame, coord_text, (px + 20, py + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw the path
        if len(self.drawing_path) > 1:
            for i in range(1, len(self.drawing_path)):
                if self.drawing_path[i - 1] is not None and self.drawing_path[i] is not None:
                    pt1 = self.drawing_path[i - 1]['absolute']
                    pt2 = self.drawing_path[i]['absolute']
                    cv2.line(display_frame, pt1, pt2, (255, 0, 255), 3)

        # Draw info panel
        self.draw_info_panel(display_frame)

        return display_frame

    def draw_info_panel(self, frame):
        """Draw information panel on the frame"""
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Number of recorded points
        cv2.putText(frame, f"Points: {len(self.drawing_path)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Instructions
        instructions = [
            "Commands:",
            "C - Clear path",
            "S - Save coordinates",
            "R - Reset paper detection",
            "Q - Quit",
            "+/- Adjust pen detection"
        ]

        y_offset = 100
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25

    def save_coordinates(self, filename=None):
        """Save the recorded coordinates to a JSON file"""
        if not self.drawing_path:
            print("No coordinates to save!")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drawing_coordinates_{timestamp}.json"

        # Prepare data for saving
        data = {
            'metadata': {
                'total_points': len(self.drawing_path),
                'timestamp': datetime.now().isoformat(),
                'paper_bounds': self.paper_bounds
            },
            'coordinates': self.drawing_path
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Coordinates saved to {filename}")
        print(f"Total points recorded: {len(self.drawing_path)}")

    def adjust_pen_color_detection(self, key):
        """Adjust pen color detection ranges with keyboard"""
        adjustment = 5

        if key == ord('+') or key == ord('='):
            self.pen_upper += adjustment
            print(f"Increased upper threshold: {self.pen_upper}")
        elif key == ord('-') or key == ord('_'):
            self.pen_lower -= adjustment
            print(f"Decreased lower threshold: {self.pen_lower}")

    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

    def run(self):
        """Main loop for the pen tracker"""
        print("=== Pen Tracker Started ===")
        print("Position your white paper in view of the camera")
        print("Use a blue-colored pen tip for tracking (adjustable)")
        print("\nControls:")
        print("  C - Clear drawing path")
        print("  S - Save coordinates to file")
        print("  R - Reset paper detection")
        print("  +/- - Adjust pen color detection")
        print("  Q - Quit")
        print("\nPress any key to continue...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Calculate FPS
            self.calculate_fps()

            # Detect paper boundaries
            self.detect_white_paper(frame)

            # Detect pen position
            self.detect_pen(frame)

            # Check if pen is in bounds
            self.pen_in_bounds = self.is_pen_in_paper_bounds()

            # Record coordinates if pen is in bounds
            self.record_coordinate()

            # Draw visualizations
            display_frame = self.draw_visualizations(frame)

            # Show the frame
            cv2.imshow('Pen Tracker', display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                self.drawing_path = []
                print("Drawing path cleared")
            elif key == ord('s'):
                self.save_coordinates()
            elif key == ord('r'):
                self.paper_bounds = None
                self.paper_contour = None
                print("Paper detection reset")
            elif key == ord('+') or key == ord('=') or key == ord('-') or key == ord('_'):
                self.adjust_pen_color_detection(key)

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

        # Ask if user wants to save before exiting
        if self.drawing_path:
            print(f"\nYou have {len(self.drawing_path)} recorded points.")
            save_option = input("Save coordinates before exiting? (y/n): ")
            if save_option.lower() == 'y':
                self.save_coordinates()


if __name__ == "__main__":
    tracker = PenTracker()
    tracker.run()
import cv2
import numpy as np
import json
from datetime import datetime
import os


class LivePenTracker:
    def __init__(self, camera_id=1):
        """Initialize live pen tracker with specified camera"""
        self.camera_id = camera_id

        # Open camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")

        # Camera properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Canvas bounds (will be detected on first frame)
        self.canvas_bounds = None
        self.canvas_detected = False

        # Trajectory data
        self.positions = []
        self.frame_count = 0

        # JSON output file
        self.json_file = "pen_positions.json"

        # Create initial JSON file
        self.save_positions_to_json()

        print("=" * 60)
        print("LIVE PEN TRACKER")
        print("=" * 60)
        print(f"Camera: {camera_id}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"JSON output: {self.json_file}")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'c' - Recalibrate canvas")
        print("  'r' - Reset trajectory")
        print("=" * 60)

    def detect_black_border_simple(self, frame, show_debug=False):
        """Simpler approach: detect the black border directly - FROM ORIGINAL CODE"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Method 1: Look for black pixels
        _, black_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        # Method 2: Edge detection
        edges = cv2.Canny(gray, 30, 100)

        # Dilate edges to connect
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)

        # Find contours from edges
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for large rectangular contours
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Must be substantial
            if area < 50000:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)

            # Check aspect ratio (should be somewhat rectangular)
            aspect = float(w) / h if h > 0 else 0
            if aspect < 0.3 or aspect > 3.0:
                continue

            # Calculate how much of the bounding box is filled
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0

            # Should be fairly filled (a rectangle would be ~1.0)
            if extent < 0.5:
                continue

            candidates.append({
                'contour': cnt,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect': aspect,
                'extent': extent
            })

        if not candidates:
            print("\n⚠️  No rectangular borders found with edge detection")
            print("   Trying alternative method...")

            # Alternative: look for the largest white region and assume border around it
            _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if white_contours:
                largest_white = max(white_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_white)

                # Assume there's a border around this white area
                border = 20  # pixels
                x += border
                y += border
                w -= 2 * border
                h -= 2 * border

                print(f"\n✓ Using white region method")
                print(f"  Canvas: {w}x{h} at ({x}, {y})")

                return (x, y, w, h)

            return None

        # Sort by area and pick largest
        candidates.sort(key=lambda x: x['area'], reverse=True)
        best = candidates[0]

        x, y, w, h = best['bbox']

        # Shrink inward to get inside the border
        border_thickness = int(min(w, h) * 0.015)  # ~1.5% of size
        x += border_thickness
        y += border_thickness
        w -= 2 * border_thickness
        h -= 2 * border_thickness

        print(f"\n✓ Canvas detected (inside black border):")
        print(f"  Area: {best['area']:.0f} pixels")
        print(f"  Border shrink: {border_thickness}px")
        print(f"  Canvas: {w}x{h} at ({x}, {y})")

        return (x, y, w, h)

    def detect_marker_pattern(self, frame, show_debug=False):
        """Detect checkerboard marker with multiple strategies - FROM ORIGINAL CODE"""
        if self.canvas_bounds is None:
            return None

        x, y, w, h = self.canvas_bounds
        canvas_region = frame[y:y + h, x:x + w].copy()

        gray_canvas = cv2.cvtColor(canvas_region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(canvas_region, cv2.COLOR_BGR2HSV)

        # Expanded color ranges to include cream/beige tones
        yellow_ranges = [
            # ("Wide Blue", np.array([90, 10, 100]), np.array([130, 150, 255])),
            ("Dark Blue", np.array([100, 80, 80]), np.array([120, 255, 180])),
            # ("Standard Blue", np.array([100, 50, 100]), np.array([130, 255, 255])),
        ]

        all_yellow_contours = []

        for i, (name, lower, upper) in enumerate(yellow_ranges):
            mask = cv2.inRange(hsv, lower, upper)

            # Clean up
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_yellow_contours.extend(contours)

        # Also try looking for any square-ish regions in general
        edges = cv2.Canny(gray_canvas, 50, 150)
        edge_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        all_contours = all_yellow_contours + list(edge_contours)

        marker_candidates = []

        for cnt in all_contours:
            area = cv2.contourArea(cnt)

            # Size filter: marker should be reasonable size
            if area < 300 or area > w * h * 0.2:  # Between 300px and 20% of canvas
                continue

            # Get bounding box
            mx, my, mw, mh = cv2.boundingRect(cnt)

            # Should be roughly square
            aspect = float(mw) / mh if mh > 0 else 0
            if aspect < 0.4 or aspect > 2.5:
                continue

            # Extract region
            if my + mh > h or mx + mw > w:
                continue

            marker_region = gray_canvas[my:my + mh, mx:mx + mw]

            if marker_region.size == 0:
                continue

            # Analyze for checkerboard pattern
            _, binary = cv2.threshold(marker_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Count black and white
            black_pixels = np.sum(binary == 0)
            white_pixels = np.sum(binary == 255)
            total = binary.size

            black_ratio = black_pixels / total
            white_ratio = white_pixels / total

            # Checkerboard should have both colors
            if black_ratio < 0.05 or white_ratio < 0.05:
                continue

            # Check for grid-like structure
            marker_edges = cv2.Canny(marker_region, 50, 150)
            edge_density = np.sum(marker_edges > 0) / marker_edges.size

            # Calculate variance (checkerboard has high variance)
            variance = np.var(marker_region)

            # Score this candidate
            score = (
                    min(black_ratio, white_ratio) * 2 +  # Both colors present
                    edge_density * 5 +  # Grid structure
                    (variance / 1000) * 0.5  # High contrast
            )

            marker_candidates.append({
                'bbox': (mx, my, mw, mh),
                'score': score,
                'area': area,
                'aspect': aspect,
                'black_ratio': black_ratio,
                'white_ratio': white_ratio,
                'edge_density': edge_density,
                'variance': variance,
                'region': marker_region,
                'binary': binary
            })

        if not marker_candidates:
            return None

        # Get best candidate
        best = max(marker_candidates, key=lambda x: x['score'])
        mx, my, mw, mh = best['bbox']

        # Calculate center
        center_x = mx + mw // 2
        center_y = my + mh // 2

        # Canvas coordinates (bottom-left origin)
        canvas_x = center_x
        canvas_y = h - center_y

        # Return marker info including bbox for visualization
        return {
            'img_center': (center_x, center_y),
            'canvas_coords': (canvas_x, canvas_y),
            'bbox': (mx, my, mw, mh),
            'score': best['score']
        }

    def draw_trajectory_canvas(self, canvas_w, canvas_h):
        """Create a canvas showing the trajectory"""
        # Create white canvas
        traj_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        if len(self.positions) < 2:
            # Show "waiting for data" message
            cv2.putText(traj_canvas, "Tracking...",
                        (canvas_w // 2 - 80, canvas_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
            return traj_canvas

        # Extract coordinates
        x_coords = [p[0] for p in self.positions]
        y_coords = [p[1] for p in self.positions]

        # Draw grid
        grid_color = (220, 220, 220)
        for i in range(0, canvas_w, 50):
            cv2.line(traj_canvas, (i, 0), (i, canvas_h), grid_color, 1)
        for i in range(0, canvas_h, 50):
            cv2.line(traj_canvas, (0, i), (canvas_w, i), grid_color, 1)

        # Draw trajectory lines with gradient color
        num_points = len(self.positions)
        for i in range(1, num_points):
            # Color gradient from blue to red
            ratio = i / num_points
            color = (
                int(255 * ratio),  # B
                0,  # G
                int(255 * (1 - ratio))  # R
            )

            pt1 = (int(x_coords[i - 1]), int(canvas_h - y_coords[i - 1]))
            pt2 = (int(x_coords[i]), int(canvas_h - y_coords[i]))

            cv2.line(traj_canvas, pt1, pt2, color, 2)

        # Draw points
        for i, (x, y) in enumerate(self.positions):
            pt = (int(x), int(canvas_h - y))

            # Color based on age
            ratio = i / num_points
            if i == 0:
                # Start point - green
                cv2.circle(traj_canvas, pt, 8, (0, 255, 0), -1)
                cv2.circle(traj_canvas, pt, 10, (0, 0, 0), 2)
            elif i == num_points - 1:
                # Current point - red
                cv2.circle(traj_canvas, pt, 8, (0, 0, 255), -1)
                cv2.circle(traj_canvas, pt, 10, (0, 0, 0), 2)
            else:
                # Trail points
                color = (
                    int(255 * ratio),
                    0,
                    int(255 * (1 - ratio))
                )
                cv2.circle(traj_canvas, pt, 3, color, -1)

        # Add info text
        cv2.putText(traj_canvas, f"Points: {len(self.positions)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        if len(self.positions) > 0:
            last_x, last_y = self.positions[-1]
            cv2.putText(traj_canvas, f"Current: ({int(last_x)}, {int(last_y)})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return traj_canvas

    def save_positions_to_json(self):
        """Save current positions to JSON file"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'frame_count': self.frame_count,
            'canvas_bounds': self.canvas_bounds,
            'total_positions': len(self.positions),
            'positions': [
                {
                    'index': i,
                    'x': int(x),
                    'y': int(y)
                }
                for i, (x, y) in enumerate(self.positions)
            ]
        }

        with open(self.json_file, 'w') as f:
            json.dump(data, f, indent=2)

    def run(self):
        """Main loop for live tracking"""
        print("\nStarting live tracking...")
        print("Waiting for first frame to detect canvas...\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            # Detect canvas on first frame
            if not self.canvas_detected:
                print("Detecting canvas...")
                self.canvas_bounds = self.detect_black_border_simple(frame, show_debug=False)

                if self.canvas_bounds is not None:
                    self.canvas_detected = True
                    print(f"✓ Canvas locked: {self.canvas_bounds}")
                    print("\nTracking started! Press 'q' to quit.\n")
                else:
                    # Show frame with message
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "Detecting canvas...",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Live Feed', display_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

            # Create annotated frame
            annotated = frame.copy()

            # Draw canvas bounds
            x, y, w, h = self.canvas_bounds
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(annotated, "FIXED Canvas", (x + 10, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Add corner markers to emphasize fixed canvas
            marker_size = 15
            # Top-left
            cv2.line(annotated, (x, y), (x + marker_size, y), (255, 0, 0), 4)
            cv2.line(annotated, (x, y), (x, y + marker_size), (255, 0, 0), 4)
            # Top-right
            cv2.line(annotated, (x + w, y), (x + w - marker_size, y), (255, 0, 0), 4)
            cv2.line(annotated, (x + w, y), (x + w, y + marker_size), (255, 0, 0), 4)
            # Bottom-left
            cv2.line(annotated, (x, y + h), (x + marker_size, y + h), (255, 0, 0), 4)
            cv2.line(annotated, (x, y + h), (x, y + h - marker_size), (255, 0, 0), 4)
            # Bottom-right
            cv2.line(annotated, (x + w, y + h), (x + w - marker_size, y + h), (255, 0, 0), 4)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - marker_size), (255, 0, 0), 4)

            # Detect marker
            marker_info = self.detect_marker_pattern(frame, show_debug=False)

            if marker_info is not None:
                # Extract marker info
                mx, my, mw, mh = marker_info['bbox']
                center_x, center_y = marker_info['img_center']
                canvas_x, canvas_y = marker_info['canvas_coords']

                # Convert to full frame coordinates
                full_mx = x + mx
                full_my = y + my
                full_center_x = x + center_x
                full_center_y = y + center_y

                # Draw bounding box
                cv2.rectangle(annotated, (full_mx, full_my),
                              (full_mx + mw, full_my + mh), (0, 255, 0), 2)

                # Draw center point
                cv2.circle(annotated, (full_center_x, full_center_y), 8, (0, 0, 255), -1)
                cv2.circle(annotated, (full_center_x, full_center_y), 12, (0, 255, 0), 2)

                # Add text info
                text = f"Marker: ({canvas_x}, {canvas_y})"
                cv2.putText(annotated, text, (full_center_x + 20, full_center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                score_text = f"Score: {marker_info['score']:.2f}"
                cv2.putText(annotated, score_text, (full_center_x + 20, full_center_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Store position
                self.positions.append((canvas_x, canvas_y))

                # Save to JSON after every new position
                self.save_positions_to_json()
            else:
                # No marker detected
                cv2.putText(annotated, "No Marker Detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Add frame counter
            cv2.putText(annotated, f"Frame: {self.frame_count}", (20, self.height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated, f"Canvas: FIXED from Frame 0", (20, self.height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Create trajectory visualization
            traj_canvas = self.draw_trajectory_canvas(w, h)

            # Resize trajectory canvas to match feed height
            traj_height = self.height
            traj_width = int(w * (traj_height / h))
            traj_canvas_resized = cv2.resize(traj_canvas, (traj_width, traj_height))

            # Combine live feed and trajectory side by side
            combined = np.hstack([annotated, traj_canvas_resized])

            # Display combined view
            cv2.imshow('Live Pen Tracker - Feed + Trajectory', combined)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✓ Quitting...")
                break
            elif key == ord('c'):
                print("\n🔄 Recalibrating canvas...")
                self.canvas_detected = False
                self.canvas_bounds = None
            elif key == ord('r'):
                print("\n🔄 Resetting trajectory...")
                self.positions = []
                self.frame_count = 0
                self.save_positions_to_json()

            self.frame_count += 1

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

        print(f"\n✓ Session complete!")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Total positions recorded: {len(self.positions)}")
        print(f"  Positions saved to: {self.json_file}")


def main():
    print("\n" + "=" * 60)
    print("LIVE MARKER TRAJECTORY TRACKER")
    print("=" * 60)

    try:
        tracker = LivePenTracker(camera_id=1)
        tracker.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
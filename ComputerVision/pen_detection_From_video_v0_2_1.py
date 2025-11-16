import cv2
import numpy as np
import json
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import requests


class TemporalFilter:
    """Simple temporal filter to smooth marker detection and reject noise"""

    def __init__(self, history_size=5, confidence_threshold=2):
        self.history = deque(maxlen=history_size)
        self.confidence_threshold = confidence_threshold
        self.last_valid_position = None
        self.max_jump_distance = 150  # Maximum pixels marker can move between frames

    def add_detection(self, position, score):
        """Add a new detection and determine if it's valid"""
        if position is None:
            self.history.append(None)
            return None

        x, y = position

        # If we have a last valid position, check if this jump is reasonable
        if self.last_valid_position is not None:
            last_x, last_y = self.last_valid_position
            distance = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)

            # Reject if jump is too large (likely noise)
            if distance > self.max_jump_distance:
                self.history.append(None)
                return self.last_valid_position  # Return predicted position instead

        # Add to history
        self.history.append((x, y, score))

        # Check if we have enough confident detections
        valid_detections = [h for h in self.history if h is not None]

        if len(valid_detections) >= self.confidence_threshold:
            # Calculate weighted average of recent positions
            weights = [h[2] for h in valid_detections]  # Use scores as weights
            positions = [(h[0], h[1]) for h in valid_detections]

            total_weight = sum(weights)
            avg_x = sum(p[0] * w for p, w in zip(positions, weights)) / total_weight
            avg_y = sum(p[1] * w for p, w in zip(positions, weights)) / total_weight

            self.last_valid_position = (int(avg_x), int(avg_y))
            return self.last_valid_position

        return None  # Not confident enough yet

    def predict_position(self):
        """Predict next position based on recent motion"""
        if len(self.history) < 2:
            return self.last_valid_position

        # Get last few valid positions
        valid_positions = [h[:2] for h in self.history if h is not None]

        if len(valid_positions) < 2:
            return self.last_valid_position

        # Simple linear prediction
        recent = valid_positions[-2:]
        dx = recent[-1][0] - recent[-2][0]
        dy = recent[-1][1] - recent[-2][1]

        if self.last_valid_position:
            pred_x = self.last_valid_position[0] + dx
            pred_y = self.last_valid_position[1] + dy
            return (int(pred_x), int(pred_y))

        return self.last_valid_position

    def reset(self):
        """Reset the filter"""
        self.history.clear()
        self.last_valid_position = None


class LivePenTracker:
    def __init__(self, camera_id=1):
        """Initialize live pen tracker with specified camera"""
        self.camera_id = camera_id
        self.executor = ThreadPoolExecutor(max_workers=10)
        

        # Buffer management
        self.MAX_FRAMES = 1000

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

        # Temporal filter for robust detection
        self.temporal_filter = TemporalFilter(history_size=5, confidence_threshold=2)

        # Debug mode
        self.debug_mode = False

        print("=" * 60)
        print("LIVE PEN TRACKER - IMPROVED BLUE MARKER DETECTION")
        print("=" * 60)
        print(f"Camera: {camera_id}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Max buffer frames: {self.MAX_FRAMES}")
        print(f"Sending to: http://localhost:6000")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'c' - Recalibrate canvas")
        print("  'r' - Reset trajectory")
        print("  'd' - Toggle debug mode")
        print("=" * 60)

    def detect_black_border_simple(self, frame, show_debug=False):
        """Detect the black border of the canvas"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection
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

    def detect_blue_marker(self, frame, show_debug=False):
        """
        Detect the BLUE SQUARE marker (not the checkerboard pattern)
        Then verify there's a pattern inside
        """
        if self.canvas_bounds is None:
            return None

        x, y, w, h = self.canvas_bounds
        canvas_region = frame[y:y + h, x:x + w].copy()

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(canvas_region, cv2.COLOR_BGR2HSV)

        # Multiple blue ranges tuned for dark blue marker
        blue_ranges = [
            # Dark saturated blue (like in your image)
            ("Dark Blue", np.array([100, 100, 80]), np.array([130, 255, 255])),
            # Slightly lighter blue
            ("Medium Blue", np.array([95, 80, 60]), np.array([125, 255, 255])),
            # Very dark blue
            ("Very Dark Blue", np.array([105, 120, 40]), np.array([125, 255, 200])),
        ]

        all_blue_masks = []

        for name, lower, upper in blue_ranges:
            mask = cv2.inRange(hsv, lower, upper)

            # Morphological operations to clean up
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            all_blue_masks.append(mask)

        # Combine all blue masks
        combined_blue_mask = np.zeros_like(all_blue_masks[0])
        for mask in all_blue_masks:
            combined_blue_mask = cv2.bitwise_or(combined_blue_mask, mask)

        # Find contours of blue regions
        contours, _ = cv2.findContours(combined_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if show_debug or self.debug_mode:
            # Show the blue mask
            cv2.imshow('Blue Mask', combined_blue_mask)

        marker_candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Size filter: blue square should be reasonable size
            if area < 500 or area > w * h * 0.3:
                continue

            # Get bounding box
            mx, my, mw, mh = cv2.boundingRect(cnt)

            # Should be roughly square (blue border is square)
            aspect = float(mw) / mh if mh > 0 else 0
            if aspect < 0.6 or aspect > 1.7:  # More lenient for square
                continue

            # Calculate solidity (how filled the shape is)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Blue square should be fairly solid
            if solidity < 0.7:
                continue

            # VERIFY: Check if there's a checkerboard pattern inside
            # Extract the center region of the blue square
            center_margin = int(min(mw, mh) * 0.2)  # 20% margin from edges
            if center_margin < 5:
                center_margin = 5

            inner_x = mx + center_margin
            inner_y = my + center_margin
            inner_w = mw - 2 * center_margin
            inner_h = mh - 2 * center_margin

            # Make sure we're within bounds
            if (inner_x < 0 or inner_y < 0 or
                    inner_x + inner_w > w or inner_y + inner_h > h or
                    inner_w <= 0 or inner_h <= 0):
                continue

            # Extract inner region
            gray_canvas = cv2.cvtColor(canvas_region, cv2.COLOR_BGR2GRAY)
            inner_region = gray_canvas[inner_y:inner_y + inner_h, inner_x:inner_x + inner_w]

            if inner_region.size == 0:
                continue

            # Check for checkerboard pattern inside
            pattern_score = self.validate_checkerboard_pattern(inner_region)

            if pattern_score < 0.3:  # Minimum pattern threshold
                continue

            # Calculate center of blue square
            center_x = mx + mw // 2
            center_y = my + mh // 2

            # Score this candidate
            aspect_score = 1.0 - abs(1.0 - aspect)  # Closer to 1.0 (square) is better
            size_score = min(area / 2000, 1.0)

            total_score = (aspect_score * 0.3 + solidity * 0.3 + pattern_score * 0.4)

            marker_candidates.append({
                'bbox': (mx, my, mw, mh),
                'center': (center_x, center_y),
                'score': total_score,
                'area': area,
                'aspect': aspect,
                'solidity': solidity,
                'pattern_score': pattern_score
            })

        if not marker_candidates:
            return None

        # Get best candidate
        best = max(marker_candidates, key=lambda x: x['score'])

        center_x, center_y = best['center']

        # Canvas coordinates (bottom-left origin)
        canvas_x = center_x
        canvas_y = h - center_y

        return {
            'img_center': (center_x, center_y),
            'canvas_coords': (canvas_x, canvas_y),
            'bbox': best['bbox'],
            'score': best['score'],
            'area': best['area'],
            'pattern_score': best['pattern_score']
        }

    def validate_checkerboard_pattern(self, region):
        """Validate that a region contains a checkerboard pattern"""
        if region.size == 0:
            return 0

        # Convert to binary
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Count black and white pixels
        black_pixels = np.sum(binary == 0)
        white_pixels = np.sum(binary == 255)
        total = binary.size

        if total == 0:
            return 0

        black_ratio = black_pixels / total
        white_ratio = white_pixels / total

        # Checkerboard should have both colors
        if black_ratio < 0.15 or black_ratio > 0.85:
            return 0
        if white_ratio < 0.15 or white_ratio > 0.85:
            return 0

        # Check for high contrast (checkerboard has high variance)
        variance = np.var(region)
        variance_score = min(variance / 2000, 1.0)

        # Check for grid structure with edges
        edges = cv2.Canny(region, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = min(edge_density * 10, 1.0)

        # Balance score (both colors should be present)
        balance_score = min(black_ratio, white_ratio) * 2.5

        # Combined pattern score
        pattern_score = (balance_score * 0.4 + variance_score * 0.3 + edge_score * 0.3)

        return pattern_score

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

    def send_position_to_server(self):
        """Send latest position to HTTP server"""
        if len(self.positions) > 0:
            data = {
                'timestamp': datetime.now().isoformat(),
                'frame_count': self.frame_count,
                'canvas_bounds': self.canvas_bounds,
                'total_positions': len(self.positions),
                'positions': [
                    {
                        'x': int(self.positions[-1][0]),
                        'y': int(self.positions[-1][1])
                    }
                ]
            }

            try:
                response = requests.post("http://localhost:6000", data=json.dumps(data))
            except Exception as e:
                # Silently fail if server not available
                pass

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

            # Add corner markers
            marker_size = 15
            cv2.line(annotated, (x, y), (x + marker_size, y), (255, 0, 0), 4)
            cv2.line(annotated, (x, y), (x, y + marker_size), (255, 0, 0), 4)
            cv2.line(annotated, (x + w, y), (x + w - marker_size, y), (255, 0, 0), 4)
            cv2.line(annotated, (x + w, y), (x + w, y + marker_size), (255, 0, 0), 4)
            cv2.line(annotated, (x, y + h), (x + marker_size, y + h), (255, 0, 0), 4)
            cv2.line(annotated, (x, y + h), (x, y + h - marker_size), (255, 0, 0), 4)
            cv2.line(annotated, (x + w, y + h), (x + w - marker_size, y + h), (255, 0, 0), 4)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - marker_size), (255, 0, 0), 4)

            # Detect marker (raw detection)
            marker_info = self.detect_blue_marker(frame, show_debug=self.debug_mode)

            # Draw prediction if available
            predicted_pos = self.temporal_filter.predict_position()
            if predicted_pos is not None:
                pred_x, pred_y = predicted_pos
                full_pred_x = x + pred_x
                full_pred_y = y + (h - pred_y)
                cv2.circle(annotated, (full_pred_x, full_pred_y), 15, (255, 165, 0), 2)

            # Process detection through temporal filter
            validated_position = None
            if marker_info is not None:
                raw_canvas_x, raw_canvas_y = marker_info['canvas_coords']
                raw_score = marker_info['score']

                # Add to temporal filter
                validated_position = self.temporal_filter.add_detection(
                    (raw_canvas_x, raw_canvas_y),
                    raw_score
                )

                # Draw raw detection (yellow - unconfirmed)
                mx, my, mw, mh = marker_info['bbox']
                center_x, center_y = marker_info['img_center']
                full_mx = x + mx
                full_my = y + my
                full_center_x = x + center_x
                full_center_y = y + center_y

                cv2.rectangle(annotated, (full_mx, full_my),
                              (full_mx + mw, full_my + mh), (0, 255, 255), 2)
                cv2.circle(annotated, (full_center_x, full_center_y), 6, (0, 255, 255), -1)

                # Show detection scores
                score_text = f"Score: {raw_score:.2f}"
                cv2.putText(annotated, score_text,
                            (full_center_x + 20, full_center_y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                # No detection, still process through filter
                validated_position = self.temporal_filter.add_detection(None, 0)

            # Draw validated position (green - confirmed)
            if validated_position is not None:
                val_x, val_y = validated_position
                full_val_x = x + val_x
                full_val_y = y + (h - val_y)

                cv2.circle(annotated, (full_val_x, full_val_y), 10, (0, 255, 0), -1)
                cv2.circle(annotated, (full_val_x, full_val_y), 14, (0, 255, 0), 3)

                text = f"Marker: ({val_x}, {val_y})"
                cv2.putText(annotated, text, (full_val_x + 20, full_val_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Store validated position
                self.positions.append(validated_position)

                # Buffer management - keep only last MAX_FRAMES positions
                if len(self.positions) > self.MAX_FRAMES:
                    self.positions = self.positions[len(self.positions) - self.MAX_FRAMES:]

                # Send to server
                self.executor.submit(self.send_position_to_server())
            else:
                # No validated marker
                cv2.putText(annotated, "No Marker Detected",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Add frame counter and status
            cv2.putText(annotated, f"Frame: {self.frame_count}", (20, self.height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated, f"Canvas: FIXED from Frame 0", (20, self.height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if self.debug_mode:
                cv2.putText(annotated, "DEBUG MODE", (self.width - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
                self.temporal_filter.reset()
            elif key == ord('r'):
                print("\n🔄 Resetting trajectory...")
                self.positions = []
                self.frame_count = 0
                self.temporal_filter.reset()
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"\n🔧 Debug mode: {'ON' if self.debug_mode else 'OFF'}")

            self.frame_count += 1

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

        print(f"\n✓ Session complete!")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Total positions recorded: {len(self.positions)}")


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
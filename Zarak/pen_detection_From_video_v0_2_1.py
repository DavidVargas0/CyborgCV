import cv2
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt


class ImprovedPenDetector:
    def __init__(self, video_path):
        self.video_path = video_path

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output directory for debug images
        self.debug_dir = "debug_output"
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

        # Create output directory for annotated frames
        self.frames_dir = "annotated_frames"
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)

        print(f"Video: {os.path.basename(video_path)}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        print(f"Total frames: {self.total_frames}")
        print(f"Debug images will be saved to: {self.debug_dir}/")
        print(f"Annotated frames will be saved to: {self.frames_dir}/")

    def save_debug_image(self, img, name):
        """Save debug image to output directory"""
        filepath = os.path.join(self.debug_dir, f"{name}.png")
        cv2.imwrite(filepath, img)
        print(f"  💾 Saved: {name}.png")

    def save_canvas_detection_image(self, frame, canvas_bounds):
        """Save visualization of detected canvas bounds"""
        vis_frame = frame.copy()
        if canvas_bounds is not None:
            x, y, w, h = canvas_bounds
            # Draw the canvas rectangle
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # Add corner markers
            marker_size = 20
            cv2.line(vis_frame, (x, y), (x + marker_size, y), (0, 255, 0), 5)
            cv2.line(vis_frame, (x, y), (x, y + marker_size), (0, 255, 0), 5)
            cv2.line(vis_frame, (x + w, y), (x + w - marker_size, y), (0, 255, 0), 5)
            cv2.line(vis_frame, (x + w, y), (x + w, y + marker_size), (0, 255, 0), 5)
            cv2.line(vis_frame, (x, y + h), (x + marker_size, y + h), (0, 255, 0), 5)
            cv2.line(vis_frame, (x, y + h), (x, y + h - marker_size), (0, 255, 0), 5)
            cv2.line(vis_frame, (x + w, y + h), (x + w - marker_size, y + h), (0, 255, 0), 5)
            cv2.line(vis_frame, (x + w, y + h), (x + w, y + h - marker_size), (0, 255, 0), 5)
            # Add text
            cv2.putText(vis_frame, "FIXED CANVAS BOUNDS", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(vis_frame, f"Position: ({x}, {y})", (x + 10, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Size: {w}x{h}", (x + 10, y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        filepath = os.path.join(self.debug_dir, "canvas_detection_fixed.png")
        cv2.imwrite(filepath, vis_frame)
        print(f"\n✓ Canvas detection saved to: canvas_detection_fixed.png")
        print(f"  This shows the FIXED canvas bounds that will be used for all frames.")

    def detect_black_border_simple(self, frame, show_debug=False):
        """Simpler approach: detect the black border directly"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if show_debug:
            self.save_debug_image(gray, "01_grayscale")

        # Method 1: Look for black pixels
        _, black_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        if show_debug:
            self.save_debug_image(black_mask, "02_black_mask")

        # Method 2: Edge detection
        edges = cv2.Canny(gray, 30, 100)

        if show_debug:
            self.save_debug_image(edges, "03_edges")

        # Dilate edges to connect
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)

        if show_debug:
            self.save_debug_image(dilated_edges, "04_dilated_edges")

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

    def detect_marker_pattern(self, frame, canvas_bounds, show_debug=False):
        """Detect checkerboard marker with multiple strategies"""
        if canvas_bounds is None:
            return None

        x, y, w, h = canvas_bounds
        canvas_region = frame[y:y + h, x:x + w].copy()

        gray_canvas = cv2.cvtColor(canvas_region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(canvas_region, cv2.COLOR_BGR2HSV)

        # Expanded color ranges to include cream/beige tones
        yellow_ranges = [
            ("Wide Yellow-Beige", np.array([5, 10, 100]), np.array([50, 150, 255])),
            ("Cream/Beige", np.array([10, 20, 150]), np.array([40, 100, 255])),
            ("Standard Yellow", np.array([20, 50, 100]), np.array([35, 255, 255])),
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

    def save_annotated_frame(self, frame, frame_idx, canvas_bounds, marker_info):
        """Save frame with annotations showing detected canvas and marker"""
        annotated = frame.copy()

        # Draw canvas bounds in blue - THIS IS FIXED FOR ALL FRAMES
        if canvas_bounds is not None:
            x, y, w, h = canvas_bounds
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(annotated, "FIXED Canvas", (x + 10, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

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

        # Draw marker detection in green
        if marker_info is not None:
            x, y, w, h = canvas_bounds
            mx, my, mw, mh = marker_info['bbox']

            # Convert to full frame coordinates
            full_mx = x + mx
            full_my = y + my

            # Draw bounding box
            cv2.rectangle(annotated, (full_mx, full_my),
                          (full_mx + mw, full_my + mh), (0, 255, 0), 2)

            # Draw center point
            center_x, center_y = marker_info['img_center']
            full_center_x = x + center_x
            full_center_y = y + center_y
            cv2.circle(annotated, (full_center_x, full_center_y), 8, (0, 0, 255), -1)
            cv2.circle(annotated, (full_center_x, full_center_y), 12, (0, 255, 0), 2)

            # Add text info
            canvas_x, canvas_y = marker_info['canvas_coords']
            text = f"Marker: ({canvas_x}, {canvas_y})"
            cv2.putText(annotated, text, (full_center_x + 20, full_center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            score_text = f"Score: {marker_info['score']:.2f}"
            cv2.putText(annotated, score_text, (full_center_x + 20, full_center_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # No marker detected
            cv2.putText(annotated, "No Marker Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Add frame number and canvas status
        cv2.putText(annotated, f"Frame: {frame_idx}", (20, self.height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, f"Canvas: FIXED from Frame 0", (20, self.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save frame
        filename = f"frame_{frame_idx:05d}.png"
        filepath = os.path.join(self.frames_dir, filename)
        cv2.imwrite(filepath, annotated)

    def process_all_frames(self):
        """Process all frames and collect marker positions"""
        print("\n" + "=" * 60)
        print("PROCESSING ALL FRAMES")
        print("=" * 60)

        # Detect canvas from first frame - THIS IS FIXED FOR ALL FRAMES
        ret, first_frame = self.cap.read()
        if not ret:
            print("❌ Cannot read first frame!")
            return None, None

        print("\n--- DETECTING CANVAS (FIRST FRAME ONLY) ---")
        print("⚠️  NOTE: Canvas will be detected ONCE and used for ALL frames")
        print("         The camera is assumed to be stationary.\n")

        canvas_bounds = self.detect_black_border_simple(first_frame, show_debug=True)

        if canvas_bounds is None:
            print("❌ Canvas detection failed!")
            return None, None

        # Save visualization of the detected canvas
        self.save_canvas_detection_image(first_frame, canvas_bounds)

        print(f"\n{'=' * 60}")
        print("CANVAS BOUNDS ARE NOW FIXED FOR ALL FRAMES")
        print(f"{'=' * 60}")
        print(f"Position: ({canvas_bounds[0]}, {canvas_bounds[1]})")
        print(f"Size: {canvas_bounds[2]}x{canvas_bounds[3]}")
        print(f"{'=' * 60}\n")

        # Reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Store all positions
        positions = []
        frame_numbers = []

        frame_idx = 0
        save_every_n = 1  # Save every frame (change to higher number to save less frequently)

        print(f"\n--- PROCESSING AND SAVING FRAMES ---")
        print(f"Saving every {save_every_n} frame(s) to {self.frames_dir}/")
        print(f"Using FIXED canvas bounds: {canvas_bounds}\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect marker - using the SAME canvas_bounds for every frame
            marker_info = self.detect_marker_pattern(frame, canvas_bounds, show_debug=False)

            if marker_info is not None:
                canvas_x, canvas_y = marker_info['canvas_coords']
                positions.append((canvas_x, canvas_y))
                frame_numbers.append(frame_idx)

            # Save annotated frame with FIXED canvas bounds
            if frame_idx % save_every_n == 0:
                self.save_annotated_frame(frame, frame_idx, canvas_bounds, marker_info)

            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx}/{self.total_frames} frames... (Found {len(positions)} markers)")

        print(f"\n✓ Processed {frame_idx} frames")
        print(f"✓ Detected marker in {len(positions)} frames ({len(positions) / frame_idx * 100:.1f}%)")
        print(f"✓ Saved {(frame_idx + save_every_n - 1) // save_every_n} annotated frames to {self.frames_dir}/")
        print(f"✓ Used FIXED canvas bounds throughout: {canvas_bounds}")

        self.cap.release()

        return canvas_bounds, positions

    def plot_trajectory(self, canvas_bounds, positions):
        """Plot the trajectory of marker positions"""
        if not positions:
            print("❌ No positions to plot!")
            return

        x, y, w, h = canvas_bounds

        # Extract x and y coordinates
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Trajectory with gradient color
        colors = np.linspace(0, 1, len(positions))
        scatter = ax1.scatter(x_coords, y_coords, c=colors, cmap='viridis', s=20, alpha=0.6)
        ax1.plot(x_coords, y_coords, 'b-', alpha=0.3, linewidth=1)
        ax1.scatter(x_coords[0], y_coords[0], c='green', s=200, marker='o', edgecolors='black', linewidths=2,
                    label='Start', zorder=5)
        ax1.scatter(x_coords[-1], y_coords[-1], c='red', s=200, marker='X', edgecolors='black', linewidths=2,
                    label='End', zorder=5)

        ax1.set_xlim(0, w)
        ax1.set_ylim(0, h)
        ax1.set_xlabel('X (pixels)', fontsize=12)
        ax1.set_ylabel('Y (pixels)', fontsize=12)
        ax1.set_title(f'Marker Trajectory ({len(positions)} points)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_aspect('equal')

        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Time progression', fontsize=10)

        # Plot 2: Heatmap
        heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=50, range=[[0, w], [0, h]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        im = ax2.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', interpolation='gaussian', aspect='auto')
        ax2.scatter(x_coords[0], y_coords[0], c='cyan', s=200, marker='o', edgecolors='white', linewidths=2,
                    label='Start', zorder=5)
        ax2.scatter(x_coords[-1], y_coords[-1], c='lime', s=200, marker='X', edgecolors='white', linewidths=2,
                    label='End', zorder=5)

        ax2.set_xlim(0, w)
        ax2.set_ylim(0, h)
        ax2.set_xlabel('X (pixels)', fontsize=12)
        ax2.set_ylabel('Y (pixels)', fontsize=12)
        ax2.set_title('Position Heatmap', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')

        cbar2 = plt.colorbar(im, ax=ax2)
        cbar2.set_label('Density', fontsize=10)

        plt.tight_layout()

        # Save plot
        output_path = os.path.join(self.debug_dir, "marker_trajectory.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Trajectory plot saved to: {output_path}")

        plt.show()


def main():
    video_path = r"C:\Users\ZKasi\OneDrive - Thornton Tomasetti, Inc\Desktop\AEC tech Hackathon\Hackathon\Cyborg-Files-20251115T195546Z-1-001\Cyborg-Files\IMG_1317.MOV"

    print("\n" + "=" * 60)
    print("MARKER TRAJECTORY TRACKER")
    print("=" * 60)

    try:
        detector = ImprovedPenDetector(video_path)
        canvas_bounds, positions = detector.process_all_frames()

        if canvas_bounds and positions:
            detector.plot_trajectory(canvas_bounds, positions)
        else:
            print("❌ Failed to process video")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
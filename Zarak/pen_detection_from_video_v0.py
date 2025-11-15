import cv2
import numpy as np
import os
from datetime import datetime


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

        # Create output directory for debug images
        self.debug_dir = "debug_output"
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

        print(f"Video: {os.path.basename(video_path)}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        print(f"Debug images will be saved to: {self.debug_dir}/")

    def save_debug_image(self, img, name):
        """Save debug image to output directory"""
        filepath = os.path.join(self.debug_dir, f"{name}.png")
        cv2.imwrite(filepath, img)
        print(f"  💾 Saved: {name}.png")

    def detect_black_border_simple(self, frame, show_debug=False):
        """Simpler approach: detect the black border directly"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if show_debug:
            self.save_debug_image(gray, "01_grayscale")
            cv2.imshow('1. Grayscale', self.resize_for_display(gray))

        # Method 1: Look for black pixels
        _, black_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        if show_debug:
            self.save_debug_image(black_mask, "02_black_mask")
            cv2.imshow('2. Black Mask (<60)', self.resize_for_display(black_mask))

        # Method 2: Edge detection
        edges = cv2.Canny(gray, 30, 100)

        if show_debug:
            self.save_debug_image(edges, "03_edges")
            cv2.imshow('3. Edges', self.resize_for_display(edges))

        # Dilate edges to connect
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)

        if show_debug:
            self.save_debug_image(dilated_edges, "04_dilated_edges")
            cv2.imshow('4. Dilated Edges', self.resize_for_display(dilated_edges))

        # Find contours from edges
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Draw all contours
        if show_debug:
            all_contours_img = frame.copy()
            cv2.drawContours(all_contours_img, contours, -1, (0, 255, 0), 2)
            self.save_debug_image(all_contours_img, "05_all_contours")
            cv2.imshow('5. All Contours', self.resize_for_display(all_contours_img))

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

        if show_debug and candidates:
            candidates_img = frame.copy()
            for i, cand in enumerate(sorted(candidates, key=lambda x: x['area'], reverse=True)[:5]):
                x, y, w, h = cand['bbox']
                color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)][i]
                cv2.rectangle(candidates_img, (x, y), (x + w, y + h), color, 3)
                cv2.putText(candidates_img,
                            f"#{i + 1}: {cand['area']:.0f}px, {w}x{h}",
                            (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)
            self.save_debug_image(candidates_img, "06_rectangle_candidates")
            cv2.imshow('6. Rectangle Candidates', self.resize_for_display(candidates_img))

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
            print("⚠️  No canvas bounds!")
            return None

        x, y, w, h = canvas_bounds
        canvas_region = frame[y:y + h, x:x + w].copy()

        if show_debug:
            self.save_debug_image(canvas_region, "07_canvas_region")
            cv2.imshow('7. Canvas Region', self.resize_for_display(canvas_region))

        gray_canvas = cv2.cvtColor(canvas_region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(canvas_region, cv2.COLOR_BGR2HSV)

        # FIXED: Expanded color ranges to include cream/beige tones
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

            if show_debug:
                mask_display = mask.copy()
                cv2.putText(mask_display, name, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                self.save_debug_image(mask_display, f"08_yellow_mask_{i}_{name.replace(' ', '_').replace('/', '_')}")
                cv2.imshow(f'8.{i} Yellow Mask: {name}', self.resize_for_display(mask_display))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_yellow_contours.extend(contours)

        # Also try looking for any square-ish regions in general
        edges = cv2.Canny(gray_canvas, 50, 150)
        edge_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if show_debug:
            edges_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.save_debug_image(edges_img, "09_canvas_edges")
            cv2.imshow('9. Canvas Edges', self.resize_for_display(edges))

        print(f"\n  Yellow/Beige regions found: {len(all_yellow_contours)}")
        print(f"  Edge contours found: {len(edge_contours)}")

        # FIXED: Properly combine contours - convert tuple to list
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

        print(f"  Marker candidates after filtering: {len(marker_candidates)}")

        if show_debug and marker_candidates:
            candidates_img = canvas_region.copy()

            # Sort by score for display
            sorted_cands = sorted(marker_candidates, key=lambda x: x['score'], reverse=True)

            for i, cand in enumerate(sorted_cands[:10]):  # Show top 10
                mx, my, mw, mh = cand['bbox']
                color = (0, 255, 0) if i == 0 else (255, 165, 0)  # Best is green, rest orange
                thickness = 3 if i == 0 else 2

                cv2.rectangle(candidates_img, (mx, my), (mx + mw, my + mh), color, thickness)
                cv2.putText(candidates_img,
                            f"#{i + 1}: {cand['score']:.2f}",
                            (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

            self.save_debug_image(candidates_img, "10_marker_candidates")
            cv2.imshow('10. Marker Candidates', self.resize_for_display(candidates_img))

        if not marker_candidates:
            print("⚠️  No marker pattern detected")
            return None

        # Get best candidate
        best = max(marker_candidates, key=lambda x: x['score'])
        mx, my, mw, mh = best['bbox']

        print(f"\n✓ Marker detected:")
        print(f"  Position: ({mx}, {my})")
        print(f"  Size: {mw}x{mh}")
        print(f"  Score: {best['score']:.3f}")
        print(f"  Black: {best['black_ratio']:.1%}, White: {best['white_ratio']:.1%}")
        print(f"  Edge density: {best['edge_density']:.3f}")
        print(f"  Variance: {best['variance']:.1f}")

        # Calculate center
        center_x = mx + mw // 2
        center_y = my + mh // 2

        # Canvas coordinates (bottom-left origin)
        canvas_x = center_x
        canvas_y = h - center_y

        print(f"  Center (image): ({center_x}, {center_y})")
        print(f"  Center (canvas): ({canvas_x}, {canvas_y})")

        if show_debug:
            # Show detailed marker view
            marker_detail = canvas_region[my:my + mh, mx:mx + mw].copy()
            cv2.line(marker_detail, (mw // 2 - 15, mh // 2), (mw // 2 + 15, mh // 2), (0, 0, 255), 2)
            cv2.line(marker_detail, (mw // 2, mh // 2 - 15), (mw // 2, mh // 2 + 15), (0, 0, 255), 2)
            cv2.circle(marker_detail, (mw // 2, mh // 2), 3, (255, 0, 0), -1)

            detail_large = cv2.resize(marker_detail, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            self.save_debug_image(detail_large, "11_marker_detail_4x")
            cv2.imshow('11. Marker Detail (4x)', detail_large)

            # Binary pattern
            binary_large = cv2.resize(best['binary'], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            self.save_debug_image(binary_large, "12_marker_binary_4x")
            cv2.imshow('12. Binary Pattern (4x)', binary_large)

        return (center_x, center_y, canvas_x, canvas_y)

    def resize_for_display(self, img, max_width=900):
        """Resize for display"""
        if len(img.shape) == 2:
            h, w = img.shape
        else:
            h, w = img.shape[:2]

        if w > max_width:
            scale = max_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(img, (new_w, new_h))
        return img

    def debug_first_frame(self):
        """Debug first frame and save all images"""
        print("\n" + "=" * 60)
        print("DETECTION DEBUG - FIRST FRAME")
        print("=" * 60)

        ret, frame = self.cap.read()
        if not ret:
            print("❌ Cannot read frame!")
            return

        # Save original
        self.save_debug_image(frame, "00_original")
        cv2.imshow('0. Original', self.resize_for_display(frame))

        # Detect canvas
        print("\n--- DETECTING CANVAS ---")
        canvas_bounds = self.detect_black_border_simple(frame, show_debug=False)

        # Detect marker
        print("\n--- DETECTING MARKER ---")
        pen_pos = self.detect_marker_pattern(frame, canvas_bounds, show_debug=False)

        # Create final result
        result = frame.copy()

        if canvas_bounds:
            x, y, w, h = canvas_bounds
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(result, f"Canvas: {w}x{h}", (x + 10, y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            if pen_pos:
                img_cx, img_cy, canvas_x, canvas_y = pen_pos
                abs_x = x + img_cx
                abs_y = y + img_cy

                # Draw marker position
                cv2.circle(result, (abs_x, abs_y), 12, (0, 0, 255), 3)
                cv2.line(result, (abs_x - 25, abs_y), (abs_x + 25, abs_y), (255, 0, 0), 3)
                cv2.line(result, (abs_x, abs_y - 25), (abs_x, abs_y + 25), (255, 0, 0), 3)
                cv2.circle(result, (abs_x, abs_y), 4, (255, 255, 0), -1)

                cv2.putText(result, f"Pen: ({canvas_x}, {canvas_y})",
                            (abs_x + 30, abs_y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        self.save_debug_image(result, "13_FINAL_RESULT")
        cv2.imshow('13. FINAL RESULT', self.resize_for_display(result))

        print("\n" + "=" * 60)
        print("✓ All debug images saved to:", self.debug_dir)
        print("=" * 60)
        print("\nPRESS ANY KEY to close windows")
        print("=" * 60)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.cap.release()


def main():
    video_path = r"C:\Users\ZKasi\OneDrive - Thornton Tomasetti, Inc\Desktop\AEC tech Hackathon\Hackathon\Cyborg-Files-20251115T195546Z-1-001\Cyborg-Files\IMG_1320.MOV"

    print("\n" + "=" * 60)
    print("IMPROVED PEN DETECTION - FIXED VERSION")
    print("=" * 60)

    try:
        detector = ImprovedPenDetector(video_path)
        detector.debug_first_frame()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
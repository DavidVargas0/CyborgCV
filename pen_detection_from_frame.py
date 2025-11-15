import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# class PenDetectionDebugger:
#     def __init__(self):
#         # self.video_path = video_path

#         if not os.path.exists(video_path):
#             raise FileNotFoundError(f"Video file not found: {video_path}")

#         self.cap = cv2.VideoCapture(video_path)
#         if not self.cap.isOpened():
#             raise ValueError(f"Cannot open video: {video_path}")

#         # Video properties
#         self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
#         self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         print(f"Video: {os.path.basename(video_path)}")
#         print(f"Resolution: {self.width}x{self.height}")
#         print(f"FPS: {self.fps}")

def detect_white_canvas(frame, show_debug=True):
    """Detect white canvas and return bounds"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgs = []
    imgs.append(gray)
    # Show grayscale
    if show_debug:
        show_image('1. Grayscale', resize_for_display(gray))

    # Threshold for white areas
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    imgs.append(thresh)
    if show_debug:
        show_image('2. White Threshold (>200)', resize_for_display(thresh))

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours
    if show_debug:
        contour_img = frame.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
        show_image('3. All White Contours', resize_for_display(contour_img))

    imgs.append(contour_img)
    if contours:
        # Sort by area
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Show top 3 largest
        # if show_debug:
        top3_img = frame.copy()
        for i, cnt in enumerate(sorted_contours[:3]):
            area = cv2.contourArea(cnt)
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i]
            cv2.drawContours(top3_img, [cnt], -1, color, 3)

            # Add label
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(top3_img, f"#{i + 1}: {area:.0f}px",
                            (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 2)

            # show_image('4. Top 3 Largest Areas', resize_for_display(top3_img))

        # Get largest
        largest = sorted_contours[0]
        area = cv2.contourArea(largest)
        x, y, w, h = cv2.boundingRect(largest)

        print(f"\nCanvas Detection:")
        print(f"  Area: {area:.0f} pixels")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {w}x{h}")

        return (x, y, w, h)

    return None

def show_image( title,img, cmap=None):
    """Display image using matplotlib"""
    plt.figure(figsize=(10, 8))
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        # Convert BGR to RGB for matplotlib
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
        else:
            plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def detect_yellow_pen(frame, canvas_bounds, show_debug=True):
    """Detect yellow pen marker"""
    if canvas_bounds is None:
        print("No canvas bounds - skipping pen detection")
        return None

    x, y, w, h = canvas_bounds

    # Extract canvas region
    canvas_region = frame[y:y + h, x:x + w].copy()

    # Convert to HSV
    hsv = cv2.cvtColor(canvas_region, cv2.COLOR_BGR2HSV)

    if show_debug:
        show_image('5. Canvas Region (BGR)', resize_for_display(canvas_region))
        show_image('6. Canvas Region (HSV)', resize_for_display(hsv))

    # Try different yellow ranges
    yellow_ranges = [
        ("Bright Yellow", np.array([20, 100, 100]), np.array([30, 255, 255])),
        ("Light Yellow", np.array([25, 50, 100]), np.array([35, 255, 255])),
        ("Wide Yellow", np.array([15, 50, 50]), np.array([40, 255, 255])),
    ]

    all_masks = []

    for name, lower, upper in yellow_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        all_masks.append((name, mask))

        if show_debug:
            # Show mask
            mask_display = mask.copy()
            cv2.putText(mask_display, name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            show_image(f'7. {name} Mask', resize_for_display(mask_display))

    # Use the first range for detection
    yellow_mask = all_masks[0][1]

    # Clean up
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask_clean = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask_clean = cv2.morphologyEx(yellow_mask_clean, cv2.MORPH_OPEN, kernel)

    if show_debug:
        show_image('8. Cleaned Yellow Mask', resize_for_display(yellow_mask_clean))

    # Find yellow contours
    contours, _ = cv2.findContours(yellow_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if show_debug:
        yellow_contours_img = canvas_region.copy()
        cv2.drawContours(yellow_contours_img, contours, -1, (0, 255, 0), 2)

        # Label each contour with area
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(yellow_contours_img, f"{area:.0f}",
                            (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 255), 2)

        show_image('9. Yellow Contours', resize_for_display(yellow_contours_img))

    if not contours:
        print("\n⚠️  No yellow regions detected!")
        print("   Try adjusting the yellow color range")
        return None

    # Get largest yellow contour
    largest_yellow = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_yellow)

    print(f"\nYellow Pen Detection:")
    print(f"  Yellow regions found: {len(contours)}")
    print(f"  Largest yellow area: {area:.0f} pixels")

    if area < 100:
        print(f"  ⚠️  Area too small (minimum: 100 pixels)")
        return None

    # Get bounding box
    mx, my, mw, mh = cv2.boundingRect(largest_yellow)
    aspect_ratio = float(mw) / mh if mh > 0 else 0

    print(f"  Bounding box: {mw}x{mh}")
    print(f"  Aspect ratio: {aspect_ratio:.2f}")

    # Calculate center
    center_x = mx + mw // 2
    center_y = my + mh // 2

    # Convert to canvas coordinates (0,0 at bottom-left)
    canvas_x = center_x
    canvas_y = h - center_y

    print(f"  Center (image coords): ({center_x}, {center_y})")
    print(f"  Center (canvas coords): ({canvas_x}, {canvas_y})")

    return (center_x, center_y, canvas_x, canvas_y)

def resize_for_display(img, max_width=800):
    """Resize image to fit on screen"""
    if len(img.shape) == 2:  # Grayscale
        h, w = img.shape
    else:  # Color
        h, w = img.shape[:2]

    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h))
    return img

def debug_first_frame(frame,show_debug):
    """Show all detections on first frame"""
    print("\n" + "=" * 60)
    print("DEBUGGING FIRST FRAME")
    print("=" * 60)

    # Read first frame
    # ret, frame = cap.read()
    # if not ret:
    #     print("Cannot read first frame!")
    #     return

    # Show original
    if show_debug:
        show_image('0. Original Frame', resize_for_display(frame))

    # Detect canvas
    print("\n--- STEP 1: CANVAS DETECTION ---")
    canvas_bounds = detect_white_canvas(frame, show_debug)

    # Detect pen
    print("\n--- STEP 2: PEN DETECTION ---")
    pen_pos = detect_yellow_pen(frame, canvas_bounds, show_debug)

    # Draw final result
    result_img = frame.copy()

    if canvas_bounds:
        x, y, w, h = canvas_bounds
        # Draw canvas boundary in green
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(result_img, f"Canvas: {w}x{h}", (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if pen_pos:
            img_cx, img_cy, canvas_x, canvas_y = pen_pos
            # Draw pen position (red circle)
            abs_x = x + img_cx
            abs_y = y + img_cy
            cv2.circle(result_img, (abs_x, abs_y), 10, (0, 0, 255), -1)
            cv2.circle(result_img, (abs_x, abs_y), 15, (255, 0, 0), 2)

            # Draw crosshair
            cv2.line(result_img, (abs_x - 20, abs_y), (abs_x + 20, abs_y), (255, 0, 0), 2)
            cv2.line(result_img, (abs_x, abs_y - 20), (abs_x, abs_y + 20), (255, 0, 0), 2)

            cv2.putText(result_img, f"Pen: ({canvas_x}, {canvas_y})",
                        (abs_x + 20, abs_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if(show_debug):
        show_image('10. FINAL RESULT', resize_for_display(result_img))

    print("\n" + "=" * 60)
    print("PRESS ANY KEY TO CLOSE")
    print("=" * 60)
    print("\nWindows shown:")
    print("  0. Original Frame")
    print("  1. Grayscale")
    print("  2. White Threshold")
    print("  3. All White Contours")
    print("  4. Top 3 Largest Areas")
    print("  5. Canvas Region (BGR)")
    print("  6. Canvas Region (HSV)")
    print("  7-8. Yellow Detection Masks")
    print("  9. Yellow Contours")
    print("  10. FINAL RESULT ⭐")
    return canvas_bounds,pen_pos
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # self.cap.release()


# def main():
# # Update this to your video path
# video_path = r"C:\Users\ZKasi\OneDrive - Thornton Tomasetti, Inc\Desktop\AEC tech Hackathon\Hackathon\Cyborg-Files-20251115T195546Z-1-001\Cyborg-Files\IMG_1321.MOV"

# print("\n" + "=" * 60)
# print("PEN DETECTION DEBUGGER")
# print("=" * 60)
# print("Shows all detection steps on first frame")
# print("=" * 60)

# try:
#     debugger = PenDetectionDebugger(video_path)
#     debugger.debug_first_frame()

# except Exception as e:
#     print(f"\n❌ Error: {e}")
#     import traceback
#     traceback.print_exc()


# if __name__ == "__main__":
# main()
import cv2
import matplotlib.pyplot as plt
from pen_detection_from_frame import * 
import importlib

# importlib.reload()

# Initialize webcam
cam = cv2.VideoCapture(1)

if not cam.isOpened():
    print("Error: Could not open camera 1")
    exit()

# Set camera properties
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create figure
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 7))
ax.axis('off')

print("Streaming... Close the window or press Ctrl+C to stop")

frame_count = 0

try:
    while True:
        ret, frame = cam.read()
        # canvas_bounds,pen_pos = debug_first_frame(frame,False)
        if not ret:
            print("Failed to capture frame")
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Clear and display new frame
        ax.clear()
        ax.imshow(frame_rgb)
        ax.axis('off')
        ax.set_title(f"Live Stream - Frame: {frame_count}")
        
        # Update display
        plt.pause(0.001)  # Small pause to update display
        
        frame_count += 1
        
        # Optional: Save specific frames
        if frame_count % 30 == 0:
            cv2.imwrite(f"frame_{frame_count:04d}.png", frame)
            print(f"Saved frame {frame_count}")
        
        # Check if window is closed
        if not plt.fignum_exists(fig.number):
            print("Window closed")
            break

except KeyboardInterrupt:
    print("\nStopped by user")

# Cleanup
cam.release()
plt.close()
print(f"Stream stopped. Total frames: {frame_count}")
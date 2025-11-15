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
        canvas_bounds,pen_pos = debug_first_frame(frame,False)

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
        # ax.plot(frame_count,100,'o')
        # ax.plot(400,400,'o')
        # Update display
        
        
        frame_count += 1
        
        # Optional: Save specific frames
        if frame_count % 30 == 0:
            cv2.imwrite(f"frame_{frame_count:04d}.png", frame)
            print(f"Saved frame {frame_count}")
        if(canvas_bounds):
            # ax.plot()
            canvasX = canvas_bounds[0]
            canvasY = canvas_bounds[1]
            canvasW = canvas_bounds[2]
            canvasH = canvas_bounds[3]
            ax.axhline(canvas_bounds[1])
            ax.axhline(canvas_bounds[1] + canvas_bounds[3])
            ax.plot(300,300,'o')
            #x,y,w,he
            ax.axvline(canvas_bounds[0])
            ax.axvline(canvas_bounds[0] + canvas_bounds[2])

            ax.plot(canvasX,canvasY,'o')
            ax.plot(canvasX + canvasW, canvasY,'o')
            ax.plot(canvasX, canvasY + canvasH,'o')
            ax.plot(canvasX + canvasW, canvasY + canvasH,'o')
            print("canvas_bounds = ",canvas_bounds[0],canvas_bounds[1],canvas_bounds[2],canvas_bounds[3])
        else:
            print("canvas boudns is None")

        # Use scatter for more control
        # ax.scatter(50,50, c='red', s=100, edgecolors='white', linewidths=2)
        if(pen_pos):
            x = pen_pos[0]
            y = pen_pos[1]
            ax.plot(x,y, 'o', color='red', markersize=15, 
                markeredgecolor='white', markeredgewidth=2)
            # ax.plot(x,y,'o')
            # Use scatter for more control
            ax.scatter(x,y, c='red', s=100, edgecolors='white', linewidths=2)
            print("pen_pos = ", pen_pos[0], pen_pos[1])
        else:
            print("pen_pos is None")
        plt.pause(0.001)  # Small pause to update display
        ax.scatter(100, 100, c='blue', s=200, marker='o')
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
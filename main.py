import cv2

# Initialize webcam (0 = default camera)
cam = cv2.VideoCapture(0)

# Capture one frame
ret, frame = cam.read()

if ret:
    cv2.imshow("Captured", frame)         
    cv2.imwrite("captured_image.png", frame)  
    cv2.waitKey(0)                      
    cv2.destroyWindow("Captured")       
else:
    print("Failed to capture image.")

cam.release()
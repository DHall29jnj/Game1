import cv2
import numpy as np
import yaml

# Define chessboard parameters
chessboard_size = (9, 6) # Number of inner corners per a chessboard row and column
square_size = 0.02 # Size of each square in meters (adjust to your board)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Loop to capture images for calibration
print("Press 'c' to capture a calibration image. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points, image points
    if ret_corners:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret_corners)

    # Display the frame
    cv2.imshow('Calibration', frame)

    key = cv2.waitKey(0)
    if key & 0xFF == ord('c') and ret_corners: # Capture if 'c' is pressed and corners are found
        print("Image captured!")
    elif key & 0xFF == ord('q'): # Quit if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()

# Calibrate the camera if images are captured
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the calibration parameters to a YAML file
    calibration_data = {
        'camera_matrix': mtx.tolist(),
        'dist_coeff': dist.tolist()
    }

    with open('calibration_params.yml', 'w') as f:
        yaml.dump(calibration_data, f)

    print("Calibration parameters saved to calibration_params.yml")
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

else:
    print("No images captured for calibration. Please capture images with the chessboard pattern.")
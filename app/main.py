import pygame
import cv2
import mediapipe as mp
import numpy as np
import sys
import yaml



# Load calibration data
calibration_file = "calibration_matrix.yaml"
camera_matrix = None
dist_coeffs = None

try:
        with open('calibration_params.yml', 'r') as f:
            calibration_data = yaml.safe_load(f)

        cam_matrix = np.array(calibration_data['camera_matrix'], dtype=np.float32)
        dist_coeffs = np.array(calibration_data['dist_coeff'], dtype=np.float32)

        print("Camera calibration parameters loaded successfully.")
        print("Camera matrix:\n", cam_matrix)
        print("Distortion coefficients:\n", dist_coeffs)

except FileNotFoundError:
        print("Error: calibration_params.yml not found. Please run the calibration script first.")
        exit()
except yaml.YAMLError as e:
        print("Error loading YAML file:", e)
        exit()

# Assume HandTracking class is defined here or imported from config_manager.py
class HandTracking:
    def __init__(self, max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
 
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
 
    def process(self, frame: np.ndarray) -> list:
        """
        Process the frame and detect hands.
        Returns a list of hand landmarks (empty if none detected).
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        return results.multi_hand_landmarks or []
 
    def draw(self, frame: np.ndarray, hands_landmarks: list):
        """
        Draw hand landmarks on the given frame.
        """
        for hand_landmarks in hands_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )

# Pygame setup
pygame.init() # Initializes Pygame modules

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) # Create the game window
pygame.display.set_caption("Hand-Controlled Game") # Set the window title

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0) # Example color for the player figure

# Player figure (a simple rectangle)
player_size = 50
player_x = SCREEN_WIDTH // 2 - player_size // 2
player_y = SCREEN_HEIGHT - player_size * 2
player_rect = pygame.Rect(player_x, player_y, player_size, player_size) # Create a rectangle to represent the player

# Hand tracking setup
hand_tracker = HandTracking()
cap = cv2.VideoCapture(0) # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open video stream.")
    pygame.quit() # Properly shut down Pygame if webcam fails
    sys.exit() # Exit the program

# Game loop
running = True # Control the game loop execution
clock = pygame.time.Clock() # To control the frame rate

while running:
    # Event handling
    for event in pygame.event.get(): # Process events like closing the window
        if event.type == pygame.QUIT: # If the user clicks the close button
            running = False # Stop the game loop

    # Get video frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break # Exit the loop if frame capture fails

    if camera_matrix is not None and dist_coeffs is not None:
            # It's usually beneficial to get the optimal new camera matrix to refine the result
            # based on a free scaling parameter. alpha=0 gives minimum unwanted pixels, alpha=1 retains all.
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
            undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)
            # Crop the image to remove black borders caused by undistortion if using alpha=0
            x, y, w_roi, h_roi = roi
            undistorted_frame = undistorted_frame[y:y+h_roi, x:x+w_roi]
    else:
            undistorted_frame = frame # Use original frame if no calibration data

    

    # Process hand tracking
    hand_landmarks_list = hand_tracker.process(frame)

    # Control player position based on hand landmarks
    if hand_landmarks_list:
        # We'll use the first hand detected
        hand_landmarks = hand_landmarks_list[0] 
        # Get the x-coordinate of the wrist (landmark 0)
        wrist_x = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x * SCREEN_WIDTH) 
        wrist_y = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y * SCREEN_WIDTH)
        
        # Update the player's vertical position
        player_y = wrist_y - player_size // 2  # Center the player on the wrist's x-coordinate
        player_rect.y = player_y
        
        # Update the player's horizontal position
        player_x = wrist_x - player_size // 2  # Center the player on the wrist's x-coordinate
        player_rect.x = player_x

    # Drawing
    screen.fill(BLACK) # Fill the background with black

    # Draw the player figure
    pygame.draw.rect(screen, RED, player_rect) # Draw the rectangle on the screen

    # (Optional) Draw hand landmarks on a separate surface or overlay
    # For simplicity, we'll skip drawing landmarks directly on the Pygame screen
    # since it involves combining OpenCV and Pygame surfaces which can be complex for a basic example.

    # Update the display
    pygame.display.flip() # Or pygame.display.update()

    # Control frame rate
    clock.tick(60) # Limit the game to 60 frames per second

# Cleanup
cap.release() # Release the webcam
pygame.quit() # Shut down Pygame
sys.exit() # Exit the program cleanly
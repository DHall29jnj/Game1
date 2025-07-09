import pygame
import cv2
import mediapipe as mp
import numpy as np
import sys


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
        results = self.hands.process(frame)
        return results.multi_hand_landmarks or []
 
    def draw(self, frame: np.ndarray, hands_landmarks: list):
        """
        Draw hand landmarks on the given frame.
        """
        for hand_landmarks in hands_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )        




    @staticmethod
    def get_instance():
        if not hasattr(HandTracking, "instance"):
            HandTracking.instance = HandTracking()
        return HandTracking.instance
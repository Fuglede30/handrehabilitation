import cv2
import mediapipe as mp
import time
import platform


class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize hand tracker with MediaPipe.
        
        Args:
            mode: Static image mode (False for video)
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum detection confidence
            tracking_confidence: Minimum tracking confidence
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def find_hands(self, img, draw=True, draw_z_values=False):
        """
        Detect hands in the image.
        
        Args:
            img: Input image (BGR format from OpenCV)
            draw: Whether to draw landmarks on the image
            draw_z_values: Whether to display z-depth values on landmarks
            
        Returns:
            Processed image with landmarks drawn
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        self.results = self.hands.process(img_rgb)
        
        # Draw hand landmarks
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Draw z-values on each landmark
                if draw_z_values:
                    h, w, c = img.shape
                    for id, landmark in enumerate(hand_landmarks.landmark):
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cz = landmark.z
                        cv2.putText(img, f"{cz:.2f}", (cx, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        return img
    
    def get_positions(self, img, hand_no=0):
        """
        Get landmark positions for a specific hand with depth.
        
        Args:
            img: Input image
            hand_no: Hand index (0 for first hand, 1 for second)
            
        Returns:
            List of landmark positions [id, x, y, z] where z is normalized depth
        """
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, landmark in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cz = landmark.z  # Normalized depth (0-1, negative = farther away)
                    landmark_list.append([id, cx, cy, cz])
        
        return landmark_list
    
    def get_hand_distance(self, img, hand_no=0):
        """
        Calculate approximate distance of hand from camera using hand size reference.
        Known reference: hand length ~12.3 cm (wrist to middle finger tip) - calibrated
        
        Args:
            img: Input image
            hand_no: Hand index (0 for first hand, 1 for second)
            
        Returns:
            Estimated distance in cm, or None if hand not detected
        """
        landmark_list = self.get_positions(img, hand_no=hand_no)
        
        if not landmark_list or len(landmark_list) < 13:
            return None
        
        # Approximate focal length for typical webcam (in pixels)
        # This varies by camera, but 800-1000 is typical
        focal_length = 900
        
        # Known hand measurements (in cm) - calibrated for this camera
        known_hand_length = 12.3  # cm (wrist to middle finger tip) - adjusted from 19cm
        
        # Get wrist (landmark 0) and middle finger tip (landmark 12)
        wrist = landmark_list[0]  # [id, x, y, z]
        middle_finger_tip = landmark_list[12]  # [id, x, y, z]
        
        # Calculate distance between wrist and finger tip in pixels
        dx = middle_finger_tip[1] - wrist[1]
        dy = middle_finger_tip[2] - wrist[2]
        measured_length_pixels = (dx**2 + dy**2) ** 0.5
        
        if measured_length_pixels < 5:  # Too small, likely noise
            return None
        
        # Using pinhole camera model: distance = (real_size * focal_length) / measured_size
        distance_cm = (known_hand_length * focal_length) / measured_length_pixels
        
        return distance_cm


def main():
    """
    Main function to capture video and track hands.
    """
    # Initialize video capture (0 is default webcam)
    # Use DirectShow on Windows to reduce latency, default on Linux/RPi
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    
    # Reduce internal buffering to minimize delay
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize hand tracker
    tracker = HandTracker(max_hands=2)
    
    # For FPS calculation
    prev_time = 0
    
    print("Starting hand tracking... Press 'q' to quit.")
    
    while True:
        # Drop older frames to keep the stream real-time
        cap.grab()
        success, img = cap.retrieve()
        
        if not success:
            print("Failed to read from camera")
            break
        
        # Find hands in the frame
        img = tracker.find_hands(img, draw_z_values=True)
        
        # Display distance of each hand from camera in upper right corner
        if tracker.results.multi_hand_landmarks:
            h, w, c = img.shape
            y_offset = 30
            for hand_idx in range(len(tracker.results.multi_hand_landmarks)):
                try:
                    distance_cm = tracker.get_hand_distance(img, hand_no=hand_idx)
                    
                    if distance_cm is not None:
                        text = f"Hand {hand_idx + 1}: {distance_cm:.1f}cm"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1.5, 1)[0]
                        x_pos = w - text_size[0] - 50
                        y_pos = y_offset + hand_idx * 25
                        cv2.putText(img, text, (x_pos, y_pos), 
                                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
                except Exception as e:
                    print(f"Error calculating distance: {e}")
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        # Display the result
        cv2.imshow("Hand Tracking", img)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

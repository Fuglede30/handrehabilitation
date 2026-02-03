import cv2
import mediapipe as mp
import time


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
        
    def find_hands(self, img, draw=True):
        """
        Detect hands in the image.
        
        Args:
            img: Input image (BGR format from OpenCV)
            draw: Whether to draw landmarks on the image
            
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
        
        return img
    
    def get_positions(self, img, hand_no=0):
        """
        Get landmark positions for a specific hand.
        
        Args:
            img: Input image
            hand_no: Hand index (0 for first hand, 1 for second)
            
        Returns:
            List of landmark positions [id, x, y]
        """
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, landmark in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmark_list.append([id, cx, cy])
        
        return landmark_list


def main():
    """
    Main function to capture video and track hands.
    """
    # Initialize video capture (0 is default webcam)
    # Use DirectShow on Windows to reduce latency
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
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
        img = tracker.find_hands(img)
        
        # Get landmark positions (optional)
        positions = tracker.get_positions(img)
        
        # Example: Print fingertip position (landmark 8 is index fingertip)
        if positions:
            # You can access specific landmarks here
            # For example, landmark 8 is the tip of the index finger
            pass
        
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

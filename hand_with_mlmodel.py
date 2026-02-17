import cv2
import mediapipe as mp
import time
import platform
import numpy as np
from ultralytics import YOLO

# ===== CONFIGURATION =====
# Select which dimension classes to detect
# Options:
#   None = detect ALL classes
#   [0, 3, 6] = detect only 2x10 (0), 2x4 (3), 3x3 (6)
#   [1, 2, 4, 5] = detect only 2x2 (1), 2x3 (2), 2x6 (4), 2x8 (5)
# Available classes:
#   0: 2x10, 1: 2x2, 2: 2x3, 3: 2x4, 4: 2x6, 5: 2x8, 6: 3x3, 7: 4x4, 8: 4x6, 9: 4x8, 10: 6x6
SELECTED_CLASSES = None  # Change this to select specific classes
# ===== END CONFIGURATION =====

# Dimension classes mapping (must match training order: sorted dimensions)
DIMENSION_CLASSES = {
    0: "2x10",
    1: "2x2",
    2: "2x3",
    3: "2x4",
    4: "2x6",
    5: "2x8",
    6: "3x3",
    7: "4x4",
    8: "4x6",
    9: "4x8",
    10: "6x6"
}

PLATE_PART_NUMBERS = {
    "11212": "3x3",
    "3020": "2x4",
    "3021": "2x3",
    "3022": "2x2",
    "3031": "4x4",
    "3032": "4x6",
    "3034": "2x8",
    "3035": "4x8",
    "3795": "2x6",
    "3832": "2x10",
    "3958": "6x6"
}



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
        
        # Clickable regions: list of (x1, y1, x2, y2, name)
        self.clickable_regions = []
        
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
    
    def add_clickable_region(self, x1, y1, x2, y2, name="Box"):
        """
        Add a clickable region (box) to track.
        
        Args:
            x1, y1: Top-left corner
            x2, y2: Bottom-right corner
            name: Label for the region
        """
        self.clickable_regions.append((x1, y1, x2, y2, name))
    
    def get_index_finger_pos(self, img, hand_no=0):
        """
        Get index finger tip position (landmark 8).
        
        Args:
            img: Input image
            hand_no: Hand index
            
        Returns:
            [id, x, y, z] or None
        """
        positions = self.get_positions(img, hand_no=hand_no)
        if positions and len(positions) > 8:
            return positions[8]  # Index finger tip
        return None
    
    def get_all_finger_tips(self, img, hand_no=0):
        """
        Get all finger tip positions.
        Finger tips are landmarks: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
        
        Args:
            img: Input image
            hand_no: Hand index
            
        Returns:
            List of finger tip positions [[id, x, y, z], ...]
        """
        positions = self.get_positions(img, hand_no=hand_no)
        finger_tip_landmarks = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_tips = []
        
        for landmark_id in finger_tip_landmarks:
            if positions and len(positions) > landmark_id:
                finger_tips.append(positions[landmark_id])
        
        return finger_tips
    
    def check_touching_region(self, img, hand_no=0):
        """
        Check which regions (if any) any finger tip is touching.
        
        Args:
            img: Input image
            hand_no: Hand index
            
        Returns:
            List of names of touched regions, or empty list if none
        """
        finger_tips = self.get_all_finger_tips(img, hand_no=hand_no)
        
        if not finger_tips:
            return []
        
        touched_regions = []
        
        # Check all finger tips against all regions
        for finger_pos in finger_tips:
            x, y = finger_pos[1], finger_pos[2]
            
            # Check all regions
            for x1, y1, x2, y2, name in self.clickable_regions:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Add region if not already in the list (avoid duplicates)
                    if name not in touched_regions:
                        touched_regions.append(name)
        
        return touched_regions
    
    def detect_squares(self, img, min_area=500, max_area=5000, canny_low=30, canny_high=100, min_aspect=0.6, max_aspect=1.7, nms_iou=0.3):
        """
        Automatically detect square/rectangular regions in the image.
        
        Args:
            img: Input image
            min_area: Minimum area threshold for detection
            max_area: Maximum area threshold for detection
            canny_low: Canny edge detection lower threshold
            canny_high: Canny edge detection upper threshold
            min_aspect: Minimum aspect ratio (width/height)
            max_aspect: Maximum aspect ratio (width/height)
            nms_iou: NMS IoU threshold
            
        Returns:
            List of detected squares (x1, y1, x2, y2, name, area, aspect_ratio)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection with configurable thresholds
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Apply morphological operations to fill gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Find contours (using integer constants instead of cv2 attributes)
        contours, _ = cv2.findContours(edges, 1, 2)  # 1 = RETR_EXTERNAL, 2 = CHAIN_APPROX_SIMPLE
        
        detected_squares = []
        
        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by area and aspect ratio (check if roughly square)
            if min_area < area < max_area:
                # Check if aspect ratio is within range
                aspect_ratio = float(w) / h if h > 0 else 0
                if min_aspect < aspect_ratio < max_aspect:
                    detected_squares.append((x, y, x + w, y + h, f"Square{len(detected_squares)}", area, aspect_ratio))
        
        # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        detected_squares = self.apply_nms(detected_squares, iou_threshold=nms_iou)
        
        return detected_squares
    
    def score_squares(self, squares, img_width, img_height):
        """
        Score squares based on detection quality (area, aspect ratio closeness to 1.0).
        
        Args:
            squares: List of detected squares
            img_width: Image width
            img_height: Image height
            
        Returns:
            List of squares sorted by score (highest first)
        """
        scored_squares = []
        
        for square in squares:
            x1, y1, x2, y2, name, area, aspect_ratio = square
            
            # Score based on:
            # 1. Area (normalized to image size)
            area_score = area / (img_width * img_height)
            
            # 2. How close aspect ratio is to 1.0 (perfect square)
            aspect_ratio_score = 1.0 - abs(aspect_ratio - 1.0)
            
            # Combined score (weighted average)
            combined_score = (area_score * 0.6) + (aspect_ratio_score * 0.4)
            
            scored_squares.append((x1, y1, x2, y2, name, combined_score))
        
        # Sort by score (descending)
        scored_squares = sorted(scored_squares, key=lambda s: s[5], reverse=True)
        
        return scored_squares
    
    def apply_nms(self, boxes, iou_threshold=0.3):
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.
        
        Args:
            boxes: List of (x1, y1, x2, y2, name, area)
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of boxes
        """
        if not boxes:
            return []
        
        # Sort by area (descending)
        boxes = sorted(boxes, key=lambda b: b[5], reverse=True)
        
        keep = []
        
        for i, box1 in enumerate(boxes):
            should_keep = True
            x1_1, y1_1, x2_1, y2_1 = box1[:4]
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            
            for kept_box in keep:
                x1_2, y1_2, x2_2, y2_2 = kept_box[:4]
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                
                # Calculate intersection
                xi1 = max(x1_1, x1_2)
                yi1 = max(y1_1, y1_2)
                xi2 = min(x2_1, x2_2)
                yi2 = min(y2_1, y2_2)
                
                intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                union = area1 + area2 - intersection
                
                iou = intersection / union if union > 0 else 0
                
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(box1)
        
        # Return keeping area and aspect_ratio for scoring
        return [(b[0], b[1], b[2], b[3], b[4], b[5], b[6]) for b in keep]


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
    
    # Load YOLO model
    print("Loading YOLO model (train_class)...")
    model = YOLO('yolo_results/train_class/weights/best.pt')
    print("YOLO model loaded!")
    
    # Convert SELECTED_CLASSES to set if not None
    if SELECTED_CLASSES is not None:
        selected_classes = set(SELECTED_CLASSES)
        selected_dims = [DIMENSION_CLASSES[id] for id in sorted(selected_classes)]
        print(f"Detecting classes: {', '.join(selected_dims)}")
    else:
        selected_classes = None
        print("Detecting ALL dimension classes")
    
    # ===== CONFIGURATION VARIABLES =====
    max_lego_bricks = 5  # Maximum number of lego bricks to detect (keeps highest confidence)
    # ===== END CONFIGURATION =====
    
    # For FPS calculation
    prev_time = 0
    
    # Track initialization time to detect lego bricks for first 5 seconds
    start_time = time.time()
    detection_complete = False
    detection_duration = 5.0  # Run YOLO for 5 seconds
    
    # Accumulate all detections during the 5 second period
    all_detections = []  # List of (x1, y1, x2, y2, confidence)
    
    print("Starting lego brick detection with YOLO... Press 'q' to quit.")
    print(f"Detecting lego bricks for the first {detection_duration} seconds...")
    
    while True:
        # Drop older frames to keep the stream real-time
        cap.grab()
        success, img = cap.retrieve()
        
        if not success:
            print("Failed to read from camera")
            break
        
        # Find hands in the frame
        img = tracker.find_hands(img, draw_z_values=False)
        
        # Use YOLO to detect lego bricks during first 5 seconds
        elapsed_time = time.time() - start_time
        if elapsed_time < detection_duration and not detection_complete:
            # Run YOLO inference
            results = model(img, verbose=False)
            
            # Extract bounding boxes from YOLO results
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    # Accumulate all detections (filtering by selected classes)
                    for box in boxes:
                        class_id = int(box.cls[0])
                        
                        # Filter by selected classes
                        if selected_classes is not None and class_id not in selected_classes:
                            continue
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        all_detections.append((float(x1), float(y1), float(x2), float(y2), conf, class_id))
                    
                    filtered_count = sum(1 for box in boxes if selected_classes is None or int(box.cls[0]) in selected_classes)
                    print(f"Detecting... ({elapsed_time:.1f}s) - Found {filtered_count} lego bricks in this frame, total {len(all_detections)} detections")
            else:
                print(f"Detecting... ({elapsed_time:.1f}s) - No lego bricks found in this frame")
        
        elif elapsed_time >= detection_duration and not detection_complete:
            detection_complete = True
            
            # Process accumulated detections to find best 5 boxes
            if all_detections:
                print(f"\nProcessing {len(all_detections)} total detections...")
                
                # Cluster similar detections using IoU threshold
                clusters = []
                iou_threshold = 0.5  # Boxes with IoU > 0.5 are considered the same object
                
                for detection in all_detections:
                    x1, y1, x2, y2, conf, class_id = detection
                    
                    # Find which cluster this detection belongs to
                    matched_cluster = None
                    for cluster in clusters:
                        # Calculate IoU with cluster's average box
                        avg_box = cluster['avg_box']
                        
                        # Calculate intersection
                        xi1 = max(x1, avg_box[0])
                        yi1 = max(y1, avg_box[1])
                        xi2 = min(x2, avg_box[2])
                        yi2 = min(y2, avg_box[3])
                        
                        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                        
                        # Calculate union
                        area1 = (x2 - x1) * (y2 - y1)
                        area2 = (avg_box[2] - avg_box[0]) * (avg_box[3] - avg_box[1])
                        union = area1 + area2 - intersection
                        
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > iou_threshold:
                            matched_cluster = cluster
                            break
                    
                    # Add to existing cluster or create new one
                    if matched_cluster:
                        matched_cluster['boxes'].append((x1, y1, x2, y2))
                        matched_cluster['confidences'].append(conf)
                        matched_cluster['class_ids'].append(class_id)
                        # Update average box
                        boxes_array = np.array(matched_cluster['boxes'])
                        matched_cluster['avg_box'] = boxes_array.mean(axis=0)
                    else:
                        clusters.append({
                            'boxes': [(x1, y1, x2, y2)],
                            'confidences': [conf],
                            'class_ids': [class_id],
                            'avg_box': np.array([x1, y1, x2, y2])
                        })
                
                # Score clusters by average confidence and count
                scored_clusters = []
                for cluster in clusters:
                    avg_conf = np.mean(cluster['confidences'])
                    count = len(cluster['boxes'])
                    # Most common class in cluster
                    most_common_class = max(set(cluster['class_ids']), key=cluster['class_ids'].count)
                    # Score combines confidence and detection frequency
                    score = avg_conf * (1 + np.log(count))  # Higher score for more frequent detections
                    scored_clusters.append((cluster['avg_box'], avg_conf, count, score, most_common_class))
                
                # Sort by score and take top N
                scored_clusters.sort(key=lambda x: x[3], reverse=True)
                top_clusters = scored_clusters[:max_lego_bricks]
                
                # Add to clickable regions
                for i, (avg_box, avg_conf, count, score, class_id) in enumerate(top_clusters):
                    x1, y1, x2, y2 = avg_box
                    dimension = DIMENSION_CLASSES.get(class_id, "Unknown")
                    region_name = f"Lego{i+1}({dimension})"
                    tracker.clickable_regions.append((int(x1), int(y1), int(x2), int(y2), region_name))
                    print(f"  Lego{i+1}: {dimension}, avg_conf={avg_conf:.2f}, detections={count}, score={score:.2f}")
                
                print(f"\nDetection complete! Fixed {len(tracker.clickable_regions)} lego bricks from {len(clusters)} total clusters.")
            else:
                print("No detections found during the 5 second period.")
            
            print("Now tracking hand touches on detected lego bricks...")
        
        # Display distance of each hand from camera in upper right corner
        if tracker.results.multi_hand_landmarks:
            h, w, c = img.shape
            y_offset = 30
            
            # Draw clickable regions (after detection is complete)
            if detection_complete:
                for x1, y1, x2, y2, name in tracker.clickable_regions:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 0), 2)
                    cv2.putText(img, name, (x1 + 5, y1 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            
            # Check touch status for each hand
            for hand_idx in range(len(tracker.results.multi_hand_landmarks)):
                try:
                    touched_regions = tracker.check_touching_region(img, hand_no=hand_idx)
                    
                    if touched_regions:
                        status = f"Hand {hand_idx + 1}: TOUCHING {', '.join(touched_regions)}"
                        color = (0, 255, 0)  # Green for touching
                    else:
                        status = f"Hand {hand_idx + 1}: Not touching"
                        color = (0, 0, 255)  # Red for not touching
                    
                    text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_PLAIN, 1.5, 1)[0]
                    x_pos = w - text_size[0] - 50
                    y_pos = y_offset + hand_idx * 25
                    cv2.putText(img, status, (x_pos, y_pos), 
                                cv2.FONT_HERSHEY_PLAIN, 1.5, color, 1)
                except Exception as e:
                    print(f"Error checking touch: {e}")
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        #cv2.putText(img, f'FPS: {int(fps)}', (10, 70), 
        #            cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
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

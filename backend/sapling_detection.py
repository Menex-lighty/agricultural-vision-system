import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import math
import csv
import os
from datetime import datetime

# Configuration - specify your paths here instead of using command line arguments
model_path = "C:/Users/rishu/Desktop/tifan/best1.pt" # Update this with your YOLO model path
csv_save_path = "C:/Users/rishu/Desktop/tifan" # Update this with your desired CSV directory path9
interval = 1 # Frame capture interval in seconds

# Create the CSV directory if it doesn't exist
os.makedirs(csv_save_path, exist_ok=True)

# Load YOLOv8 model with keypoints
model = YOLO(model_path)

# Constants
ANGLE_MIN = 60
ANGLE_MAX = 120
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Temporal smoothing parameters
SMOOTHING_WINDOW = 5
angle_history = []

# Sapling numbering counters
left_counter = 1 # Start with even for left side
right_counter = 2 # Start with odd for right side

# CSV logging setup
csv_filename = os.path.join(csv_save_path, f"sapling_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
csv_header = ['timestamp', 'frame_number', 'sapling_id', 'plantation_status', 'angle', 'confidence', 'position']

# Create CSV file with header
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_header)

# Track already logged sapling IDs to avoid duplicates
logged_saplings = set()

# Initialize webcam
print("Opening webcam")
cap = cv2.VideoCapture(1) # Use 0 for default webcam, change if you have multiple cameras
# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Webcam properties: {frame_width}x{frame_height} at {fps} FPS")

# Calculate frames to skip based on interval and FPS
frames_per_interval = int(fps * interval)
print(f"Processing one frame every {interval} seconds ({frames_per_interval} frames)")

# Performance tracking
frame_times = []
detection_times = []
processing_times = []

# Visualization settings - NEW
VIS_SETTINGS = {
    'keypoint_size': 5,                           # Size of keypoint markers
    'keypoint_colors': [                          # Different colors for each keypoint
        (255, 0, 0),      # Blue - point 0 (top)
        (0, 255, 255),    # Yellow - point 1
        (0, 255, 0),      # Green - point 2
        (255, 0, 255),    # Magenta - point 3 (bottom)
    ],
    'box_border_thickness': 2,                    # Thickness of bounding box border
    'box_alpha': 0.3,                             # Transparency for box fill (0-1)
    'show_keypoint_labels': True,                 # Show keypoint numbers
    'show_skeleton': True,                        # Show lines connecting keypoints
    'font_size': 0.5,                             # Text size
    'font_thickness': 2,                          # Text thickness
    'proper_color': (0, 255, 0),                  # Green for proper planting
    'tilted_color': (0, 0, 255),                  # Red for tilted planting
    'show_legend': True,                          # Show explanation of colors/markers
}

def to_numpy(tensor):
    """Convert PyTorch tensor to numpy array safely"""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return tensor

def calculate_orientation(keypoints):
    """
    Calculate orientation from keypoints with weighted angles
    Gives more weight to the bottom part of the stem
    """
    if keypoints is None or len(keypoints) < 4:
        return None, None, None
    
    # Define weights for different segments (more weight to bottom segments)
    weights = [0.2, 0.3, 0.5] # From top to bottom
    segment_angles = []
    segment_lines = []
    
    # Calculate angle for each segment
    for i in range(3):
        top = keypoints[i]
        bottom = keypoints[i+1]
        
        # Convert to numpy safely if needed
        top_np = to_numpy(top)
        bottom_np = to_numpy(bottom)
        
        # Check for invalid keypoints
        if (top_np[0] == 0 and top_np[1] == 0) or (bottom_np[0] == 0 and bottom_np[1] == 0) or \
           np.isnan(top_np).any() or np.isnan(bottom_np).any():
            continue
        
        # Calculate angle
        dx = bottom_np[0] - top_np[0]
        dy = top_np[1] - bottom_np[1] # Flipped for image coordinates
        
        if dx == 0 and dy == 0: # Avoid division by zero
            continue
            
        angle_rad = math.atan2(dy, dx)
        angle_deg = abs(math.degrees(angle_rad))
        
        segment_angles.append((angle_deg, weights[i]))
        segment_lines.append(((int(top_np[0]), int(top_np[1])), (int(bottom_np[0]), int(bottom_np[1]))))
    
    if not segment_angles:
        return None, None, None
    
    # Calculate weighted average angle
    total_weight = sum(weight for _, weight in segment_angles)
    if total_weight == 0:
        return None, None, None
    
    weighted_angle = sum(angle * weight for angle, weight in segment_angles) / total_weight
    
    # For visualization, return the main stem line (points 0 to 3)
    if len(keypoints) >= 4:
        kp0_np = to_numpy(keypoints[0])
        kp3_np = to_numpy(keypoints[3])
        main_line = (int(kp0_np[0]), int(kp0_np[1])), (int(kp3_np[0]), int(kp3_np[1]))
    else:
        main_line = None
    
    return weighted_angle, main_line, segment_lines

def apply_smoothing(new_angle):
    """Apply temporal smoothing to angle measurements"""
    global angle_history
    
    if new_angle is None:
        return None
        
    angle_history.append(new_angle)
    if len(angle_history) > SMOOTHING_WINDOW:
        angle_history.pop(0)
        
    # Filter out None values
    valid_angles = [a for a in angle_history if a is not None]
    
    if not valid_angles:
        return None
        
    # Return median for robustness against outliers
    return float(np.median(valid_angles))

def classify_orientation(angle):
    """Classify sapling orientation based on angle"""
    if angle is None:
        return "Unknown"
        
    if ANGLE_MIN <= angle <= ANGLE_MAX:
        return "Properly Planted"
    else:
        return "Tilted"

def get_sapling_id_for_position(x, frame_width):
    """Determine sapling ID based on position in frame"""
    global left_counter, right_counter
    
    # Determine if sapling is in left or right half
    if x < frame_width / 2: # Left half
        sapling_id = left_counter
        left_counter += 2 # Increment by 2 to maintain even numbers
    else: # Right half
        sapling_id = right_counter
        right_counter += 2 # Increment by 2 to maintain odd numbers
    
    return sapling_id

def log_sapling_data(frame_number, sapling_id, status, angle, confidence, position):
    """Log sapling data to CSV file"""
    global logged_saplings
    
    # For video processing, we track frame number with sapling ID
    unique_id = f"{sapling_id}_{frame_number}"
    
    # Only log each sapling ID once per frame
    if unique_id in logged_saplings:
        return
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, frame_number, sapling_id, status, 
                        f"{angle:.2f}" if angle else "Unknown", 
                        f"{confidence:.2f}" if confidence else "Unknown", position])
    
    logged_saplings.add(unique_id)
    print(f"Frame {frame_number}: Logged sapling {sapling_id} ({status}) to CSV")

# NEW FUNCTION - Draw enhanced bounding box with fill
def draw_bounding_box(img, x, y, w, h, color, alpha=0.3, thickness=2):
    """Draw a bounding box with semi-transparent fill"""
    # Draw the filled rectangle with transparency
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)  # Filled rectangle
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw the border rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)  # Border only

# NEW FUNCTION - Draw legend
def draw_legend(img):
    """Draw legend explaining the markers and colors"""
    # Set up legend position and dimensions
    legend_x = 10
    legend_y = frame_height - 180
    legend_w = 200
    legend_h = 170
    padding = 10
    line_height = 20
    
    # Draw semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Draw border
    cv2.rectangle(img, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h), (255, 255, 255), 1)
    
    # Add title
    cv2.putText(img, "LEGEND", (legend_x + padding, legend_y + line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add keypoint explanations
    y_offset = legend_y + 2*line_height
    for i, color in enumerate(VIS_SETTINGS['keypoint_colors']):
        if i < 4:  # Only show for the 4 keypoints we use
            # Draw marker example
            marker_x = legend_x + padding + 5
            marker_y = y_offset + i*line_height
            cv2.circle(img, (marker_x, marker_y), VIS_SETTINGS['keypoint_size'], color, -1)
            
            # Add label
            cv2.putText(img, f"Keypoint {i}", (marker_x + 15, marker_y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add status explanations
    y_offset += 5*line_height
    
    # Properly planted
    status_x = legend_x + padding
    cv2.rectangle(img, (status_x, y_offset), (status_x + 15, y_offset + 15), VIS_SETTINGS['proper_color'], -1)
    cv2.putText(img, "Properly Planted", (status_x + 20, y_offset + 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Tilted
    y_offset += line_height
    cv2.rectangle(img, (status_x, y_offset), (status_x + 15, y_offset + 15), VIS_SETTINGS['tilted_color'], -1)
    cv2.putText(img, "Tilted", (status_x + 20, y_offset + 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Initialize frame counter
frame_count = 0
last_processed_time = 0
saplings_in_frame = 0

# Add a window for displaying the video
cv2.namedWindow("Sapling Detection", cv2.WINDOW_NORMAL)

# Process the video
try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break
        
        frame_count += 1
        should_process = (frame_count % frames_per_interval == 1)
        annotated_frame = frame.copy()
        center_x = frame_width // 2
        
        # Draw center dividing line - NEW
        cv2.line(annotated_frame, (center_x, 0), (center_x, frame_height), (100, 100, 100), 1, cv2.LINE_AA)
        
        # Reset detection flags for each frame
        left_detected = False
        right_detected = False
        saplings_in_frame = 0
        
        if should_process:
            start_time = time.time()
            detection_start = time.time()
            results = model.predict(source=frame, show=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            detection_end = time.time()
            
            processing_start = time.time()
            
            if results and len(results) > 0:
                keypoints = results[0].keypoints
                
                if keypoints is not None and len(keypoints.xy) > 0:
                    # Process each detected object
                    for i, kps in enumerate(keypoints.xy):
                        # Skip if not enough keypoints
                        if len(kps) < 4:
                            continue
                        
                        saplings_in_frame += 1
                        
                        # Calculate orientation with weighted segments
                        angle, main_line, segment_lines = calculate_orientation(kps)
                        
                        # Apply temporal smoothing
                        smoothed_angle = apply_smoothing(angle)
                        
                        # Classify orientation
                        classification = classify_orientation(smoothed_angle)
                        
                        # Skip drawing if angle couldn't be calculated
                        if angle is None or main_line is None:
                            continue
                            
                        # Calculate center point of sapling
                        kps_np = to_numpy(kps)
                        center_point_x = (kps_np[0][0] + kps_np[3][0]) / 2
                        
                        # Determine if this is in left or right half
                        is_left = center_point_x < center_x
                        position = "Left" if is_left else "Right"
                        
                        # Update detection flags
                        if is_left:
                            left_detected = True
                        else:
                            right_detected = True
                        
                        # Get sapling ID
                        sapling_id = get_sapling_id_for_position(center_point_x, frame_width)
                            
                        # Get confidence if available
                        confidence = None
                        if hasattr(results[0], 'boxes') and len(results[0].boxes) > i:
                            confidence = results[0].boxes.conf[i].item()
                            
                        # Log sapling data to CSV
                        log_sapling_data(frame_count, sapling_id, classification, smoothed_angle, confidence, position)
                            
                        # Draw box around sapling (using first and last keypoint)
                        top_point = (int(kps_np[0][0]), int(kps_np[0][1]))
                        bottom_point = (int(kps_np[3][0]), int(kps_np[3][1]))
                            
                        # Calculate box dimensions
                        box_width = max(50, int(abs(kps_np[1][0] - kps_np[2][0]) * 2))
                        box_height = int(abs(bottom_point[1] - top_point[1]) * 1.1)
                        box_x = int(top_point[0] - box_width // 2)
                        box_y = int(top_point[1])
                        
                        # Choose color based on classification
                        box_color = VIS_SETTINGS['proper_color'] if classification == "Properly Planted" else VIS_SETTINGS['tilted_color']
                            
                        # Draw enhanced bounding box with fill - IMPROVED
                        draw_bounding_box(
                            annotated_frame, 
                            box_x, box_y, 
                            box_width, box_height, 
                            box_color, 
                            VIS_SETTINGS['box_alpha'], 
                            VIS_SETTINGS['box_border_thickness']
                        )
                            
                        # Draw main stem line
                        cv2.line(annotated_frame, main_line[0], main_line[1], (255, 255, 0), 2, cv2.LINE_AA)
                            
                        # Draw segment lines with different colors based on weight
                        colors = [(100, 100, 255), (100, 255, 100), (255, 100, 100)] # Different color for each segment
                            
                        if VIS_SETTINGS['show_skeleton']:
                            for j, line in enumerate(segment_lines):
                                if j < len(colors):
                                    cv2.line(annotated_frame, line[0], line[1], colors[j], 1, cv2.LINE_AA)
                            
                        # Draw enhanced keypoints - IMPROVED
                        for idx, (x, y) in enumerate(kps_np):
                            # Use different colors for each keypoint
                            keypoint_color = VIS_SETTINGS['keypoint_colors'][idx] if idx < len(VIS_SETTINGS['keypoint_colors']) else (0, 0, 255)
                            
                            # Draw outer circle (white border)
                            cv2.circle(annotated_frame, 
                                      (int(x), int(y)), 
                                      VIS_SETTINGS['keypoint_size'] + 2, 
                                      (255, 255, 255), 
                                      -1)
                                      
                            # Draw inner circle (colored)
                            cv2.circle(annotated_frame, 
                                      (int(x), int(y)), 
                                      VIS_SETTINGS['keypoint_size'], 
                                      keypoint_color, 
                                      -1)
                            
                            # Add keypoint number
                            if VIS_SETTINGS['show_keypoint_labels']:
                                cv2.putText(annotated_frame, 
                                           str(idx), 
                                           (int(x)+5, int(y)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.4, 
                                           (255, 255, 255), 
                                           1)
                            
                        # Display angle
                        mid_x = int((top_point[0] + bottom_point[0]) / 2)
                        mid_y = int((top_point[1] + bottom_point[1]) / 2)
                            
                        # Add a background for text - NEW
                        text_size = cv2.getTextSize(f"Angle: {smoothed_angle:.1f}°", cv2.FONT_HERSHEY_SIMPLEX, 
                                                VIS_SETTINGS['font_size'], VIS_SETTINGS['font_thickness'])[0]
                        cv2.rectangle(annotated_frame, 
                                    (mid_x - 5, mid_y - 5), 
                                    (mid_x + text_size[0] + 5, mid_y + text_size[1] + 5), 
                                    (0, 0, 0), 
                                    -1)
                            
                        # Highlight smoothed vs raw angle if different
                        angle_text = f"Angle: {smoothed_angle:.1f}° (raw: {angle:.1f}°)" if abs(smoothed_angle - angle) > 2 else f"Angle: {smoothed_angle:.1f}°"
                        cv2.putText(annotated_frame, 
                                   angle_text, 
                                   (mid_x, mid_y + text_size[1]),
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   VIS_SETTINGS['font_size'], 
                                   (0, 255, 255), 
                                   VIS_SETTINGS['font_thickness'])
                            
                        # Display sapling ID with better visibility - IMPROVED
                        # Background for sapling ID
                        id_text = f"ID: {sapling_id}"
                        id_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, 
                                    (box_x, box_y - 65), 
                                    (box_x + id_size[0] + 10, box_y - 45), 
                                    (0, 0, 0), 
                                    -1)
                        cv2.putText(annotated_frame, 
                                   id_text, 
                                   (box_x + 5, box_y - 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, 
                                   (255, 255, 0), 
                                   2)
                            
                        # Display classification result with better visibility - IMPROVED
                        # Background for classification
                        class_size = cv2.getTextSize(classification, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, 
                                    (box_x, box_y - 30), 
                                    (box_x + class_size[0] + 10, box_y - 10), 
                                    (0, 0, 0), 
                                    -1)
                        result_color = VIS_SETTINGS['proper_color'] if classification == "Properly Planted" else VIS_SETTINGS['tilted_color']
                        cv2.putText(annotated_frame, 
                                   classification, 
                                   (box_x + 5, box_y - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, 
                                   result_color, 
                                   2)
                            
                        # Display confidence if available - IMPROVED
                        if confidence is not None:
                            # Background for confidence
                            conf_text = f"Conf: {confidence:.2f}"
                            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(annotated_frame, 
                                        (box_x, box_y - 100), 
                                        (box_x + conf_size[0] + 10, box_y - 80), 
                                        (0, 0, 0), 
                                        -1)
                            cv2.putText(annotated_frame, 
                                      conf_text, 
                                      (box_x + 5, box_y - 85),
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, 
                                      (255, 255, 255), 
                                      2)
            
                # Update counters if needed for next frame
                if left_detected:
                    # Next frame should have next even number for right
                    right_counter = (left_counter - 2) + 1
                elif right_detected and not left_detected:
                    # Next frame should have next even number for left
                    left_counter = (right_counter - 1) + 1
                    
            processing_end = time.time()
                    
            # Calculate performance metrics
            frame_time = time.time() - start_time
            detection_time = detection_end - detection_start
            processing_time = processing_end - processing_start
                    
            frame_times.append(frame_time)
            detection_times.append(detection_time)
            processing_times.append(processing_time)
                    
            # Only keep last 30 measurements
            if len(frame_times) > 30:
                frame_times.pop(0)
                detection_times.pop(0)
                processing_times.pop(0)
        else:
            # Add a note to frame when skipping processing
            cv2.putText(annotated_frame, "SKIPPED FRAME", (frame_width // 2 - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Calculate averages
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
        avg_detection = sum(detection_times) / len(detection_times) * 1000 if detection_times else 0
        avg_processing = sum(processing_times) / len(processing_times) * 1000 if processing_times else 0
        
        # Create info panel with semi-transparent background - IMPROVED
        panel_width = 220
        panel_height = 140
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (5, 5), (5 + panel_width, 5 + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        # Display performance info with improved visibility
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display counter status
        cv2.putText(annotated_frame, f"Next ID: L={left_counter} R={right_counter}", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Saplings in frame: {saplings_in_frame}", (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Indicate if this frame was processed
        process_status = "PROCESSED" if should_process else "SKIPPED"
        cv2.putText(annotated_frame, f"Frame status: {process_status}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0) if should_process else (100, 100, 255), 2)
        
        # Show the CSV file path being written to
        cv2.putText(annotated_frame, f"CSV: {os.path.basename(csv_filename)}", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw legend if enabled - NEW
        if VIS_SETTINGS['show_legend']:
            draw_legend(annotated_frame)
        
        # Show result
        cv2.imshow("Sapling Detection", annotated_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print("Processing interrupted by user")
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Print final performance summary
    print("\n--- Final Performance Summary ---")
    print(f"Processed {frame_count} frames")
    print(f"Processed one frame every {interval} seconds")
    if frame_times:
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
        print(f"Average FPS: {avg_fps:.2f}")
    if detection_times:
        avg_detection = sum(detection_times) / len(detection_times) * 1000
        print(f"Average detection time: {avg_detection:.2f}ms")
    if processing_times:
        avg_processing = sum(processing_times) / len(processing_times) * 1000
        print(f"Average processing time: {avg_processing:.2f}ms")
    print(f"Total unique saplings logged: {len(set([id.split('_')[0] for id in logged_saplings]))}")
    print(f"CSV data saved to: {os.path.abspath(csv_filename)}")
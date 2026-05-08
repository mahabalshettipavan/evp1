import cv2
from ultralytics import YOLO
import os
import sys

def detect_emergency_vehicle(source, model_path="yolov8m.pt"):
    """
    Runs YOLO inference on a video file or webcam to detect emergency vehicles.
    
    Args:
        source: Path to video file or '0' for webcam.
        model_path: Path to the YOLO model file.
    """
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model = YOLO(model_path)
    
    # Open video source
    if source == '0':
        print("Opening Webcam...")
        cap = cv2.VideoCapture(0)
    else:
        print(f"Opening video file: {source}")
        if not os.path.exists(source):
            print(f"Error: Video file not found at {source}")
            return
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame.")
            break

        # Run inference
        # We look for classes that might be emergency vehicles. 
        # COCO dataset (default YOLO) has 'car', 'truck', 'bus'. 
        # It does NOT have 'ambulance' by default unless trained on a custom dataset.
        # If your model is custom trained for 'ambulance', it will work.
        # If it is standard YOLOv8, it will detect 'truck' or 'car'.
        
        results = model(frame, verbose=False, conf=0.5)
        
        # Visualize results on the frame
        # annotator = results[0].plot() # Default plot
        
        # Custom plotting to highlight emergency vehicles
        frame_out = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Color coding: Red for emergency, Green for others
                if label in ['ambulance', 'firetruck', 'police']:
                    color = (0, 0, 255) # Red
                    text = f"EMERGENCY: {label} {conf:.2f}"
                    # Draw thicker box
                    cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 3)
                    # Add alert text
                    cv2.putText(frame_out, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    print(f"!!! {label.upper()} DETECTED !!!")
                else:
                    color = (0, 255, 0) # Green
                    cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_out, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Real Life Scenario Test", frame_out)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Default to the custom trained model if it exists
    custom_model_path = r"C:\Users\PAVAN\Desktop\MINIPROJECT\runs\detect\yolov8s_custom_training3\weights\best.pt"
    if os.path.exists(custom_model_path):
        model_file = custom_model_path
        print(f"Using custom trained model: {model_file}")
    else:
        model_file = "yolov8s.pt"
        print(f"Custom model not found. Using standard model: {model_file}")
        print("WARNING: Standard model may not detect 'ambulance' correctly.")
    
    # Check if user provided arguments
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
    else:
        # Ask user for input if not provided via command line
        print("--- Real Life Scenario Test ---")
        print("1. Test with Webcam")
        print("2. Test with Video File")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            video_source = '0'
        elif choice == '2':
            video_source = input("Enter path to video file (e.g., traffic.mp4): ").strip()
            # Remove quotes if user added them
            video_source = video_source.strip('"').strip("'")
        else:
            print("Invalid choice. Exiting.")
            sys.exit()

    detect_emergency_vehicle(video_source, model_file)

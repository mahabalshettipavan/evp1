import cv2
from ultralytics import YOLO
import os
import sys

def detect_emergency_vehicle(source, model_path="yolov8m.pt"):
    """
    Runs YOLO inference on a video file or webcam to detect emergency vehicles.
    """
    
    # --- STEP 1: LOAD MODEL FIRST ---
    # We must load the model before we can check its class names
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model = YOLO(model_path)
    
    # Debug: Print classes to confirm what the model detects
    print(f"Model classes: {model.names}") 

    # --- STEP 2: OPEN VIDEO SOURCE ---
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

        # --- STEP 3: INFERENCE WITH FILTERS ---
        # FIX A: conf=0.5 ignores weak detections (removes noise)
        results = model(frame, verbose=False, conf=0.5) 
        
        frame_out = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # --- STEP 4: STRICT CLASS CHECKING ---
                # Only highlight specific emergency classes
                emergency_classes = ['ambulance', 'firetruck', 'police']
                
                if label in emergency_classes:
                    color = (0, 0, 255) # Red for Emergency
                    text = f"EMERGENCY: {label} {conf:.2f}"
                    
                    # Draw thicker box
                    cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 3)
                    # Add alert text
                    cv2.putText(frame_out, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    print(f"!!! {label.upper()} DETECTED !!!")
                    
                # Optional: Draw Green box for normal traffic if detected
                elif label in ['car', 'truck', 'bus']:
                    color = (0, 255, 0) # Green for others
                    cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_out, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Real Life Scenario Test", frame_out)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to your custom model
    custom_model_path = r"C:\Users\PAVAN\Desktop\MINIPROJECT\runs\detect\yolov8s_custom_training3\weights\best.pt"
    
    if os.path.exists(custom_model_path):
        model_file = custom_model_path
        print(f"Using custom trained model: {model_file}")
    else:
        model_file = "yolov8s.pt"
        print(f"Custom model not found. Using standard model: {model_file}")
    
    # Check for command line args or ask user
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
    else:
        print("--- Real Life Scenario Test ---")
        print("1. Test with Webcam")
        print("2. Test with Video File")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            video_source = '0'
        elif choice == '2':
            video_source = input("Enter path to video file (e.g., traffic.mp4): ").strip()
            # Clean up path quotes
            video_source = video_source.strip('"').strip("'")
        else:
            print("Invalid choice. Exiting.")
            sys.exit()

    detect_emergency_vehicle(video_source, model_file)
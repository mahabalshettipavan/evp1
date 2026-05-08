import cv2
from ultralytics import YOLO
import os
import sys
from deep_sort_realtime.deepsort_tracker import DeepSort

# Try importing the YouTube helper. If not found, we handle it gracefully later.
try:
    from cap_from_youtube import cap_from_youtube
    YOUTUBE_SUPPORT = True
except ImportError:
    YOUTUBE_SUPPORT = False

def detect_emergency_vehicle(source, model_path="yolov8m.pt"):
    """
    Runs YOLO inference on a video file, webcam, or YouTube URL.
    Integrates DeepSORT for object tracking.
    """
    
    # --- STEP 1: LOAD MODEL FIRST ---
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model = YOLO(model_path)
    
    # Initialize DeepSORT Tracker
    tracker = DeepSort(max_age=30, n_init=3)
    
    # Debug: Print classes to confirm what the model detects
    print(f"Model classes: {model.names}") 

    # --- STEP 2: OPEN VIDEO SOURCE ---
    cap = None
    
    # CASE A: Webcam
    if source == '0':
        print("Opening Webcam...")
        cap = cv2.VideoCapture(0)
        
    # CASE B: YouTube URL (Check for http/https)
    elif source.startswith(('http://', 'https://')):
        if YOUTUBE_SUPPORT:
            print(f"Attempting to load YouTube stream: {source}")
            # resolution='best' or '720p' are common choices
            cap = cap_from_youtube(source, resolution='720p') 
        else:
            print("Error: You selected a YouTube link but 'cap_from_youtube' is not installed.")
            print("Please run: pip install cap-from-youtube")
            return

    # CASE C: Local Video File
    else:
        print(f"Opening video file: {source}")
        if not os.path.exists(source):
            print(f"Error: Video file not found at {source}")
            return
        cap = cv2.VideoCapture(source)

    if cap is None or not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame.")
            break

        # --- STEP 3: INFERENCE WITH FILTERS ---
        results = model(frame, verbose=False, conf=0.72) 
        
        frame_out = frame.copy()
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                
                # Filter for relevant classes
                emergency_classes = ['ambulance', 'firetruck', 'police']
                other_classes = ['car', 'truck', 'bus']
                
                if label in emergency_classes or label in other_classes:
                    # DeepSORT expects [left, top, w, h]
                    detections.append(([x1, y1, w, h], conf, label))

        # --- STEP 4: UPDATE TRACKER ---
        tracks = tracker.update_tracks(detections, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Get the class from the track (DeepSORT stores it if passed in detections)
            label = track.det_class
            
            # --- STEP 5: DRAWING LOGIC ---
            emergency_classes = ['ambulance', 'firetruck', 'police']
            
            if label in emergency_classes:
                color = (0, 0, 255) # Red for Emergency
                text = f"ID: {track_id} EMERGENCY: {label}"
                
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame_out, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
            elif label in ['car', 'truck', 'bus']:
                color = (0, 255, 0) # Green for others
                text = f"ID: {track_id} {label}"
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_out, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
        print("3. Test with YouTube URL")
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            video_source = '0'
        elif choice == '2':
            video_source = input("Enter path to video file: ").strip().strip('"').strip("'")
        elif choice == '3':
            video_source = input("Enter YouTube URL: ").strip()
        else:
            print("Invalid choice. Exiting.")
            sys.exit()

    detect_emergency_vehicle(video_source, model_file)
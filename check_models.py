import os
from ultralytics import YOLO
import cv2
import numpy as np

def check_model(model_path):
    print(f"Checking model: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: File not found at {model_path}")
        return False
    
    try:
        model = YOLO(model_path)
        # Create a dummy image (black image)
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(img)
        
        print(f"Successfully loaded and ran inference on {model_path}")
        # Print some info about the results to be sure
        for r in results:
            print(f"Detected {len(r.boxes)} objects (on dummy image)")
            
        return True
    except Exception as e:
        print(f"Failed to load or run {model_path}. Error: {e}")
        return False

if __name__ == "__main__":
    base_path = os.getcwd()
    model_1 = os.path.join(base_path, "yolo11n.pt")
    model_2 = os.path.join(base_path, "yolov8s.pt")
    
    success_1 = check_model(model_1)
    print("-" * 20)
    success_2 = check_model(model_2)
    
    if success_1 and success_2:
        print("\nBOTH MODELS ARE WORKING CORRECTLY.")
    else:
        print("\nSOME MODELS FAILED.")

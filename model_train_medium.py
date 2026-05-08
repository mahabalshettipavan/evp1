from ultralytics import YOLO
import os

def main():
    # 1. Load the Medium model
    # 'yolov8m.pt' will be automatically downloaded if not present
    print("Loading YOLOv8m (Medium) model...")
    model = YOLO('yolov8m.pt')  

    # 2. Define the path to the data.yaml file
    # Ensure this path is correct relative to where you run the script
    data_path = os.path.abspath(os.path.join('unified_dataset', 'data.yaml'))

    if not os.path.exists(data_path):
        print(f"Error: data.yaml not found at {data_path}")
        print("Please run merge_dataset.py first to generate the dataset.")
        return

    # 3. Train the model
    print("Starting training with YOLOv8m...")
    results = model.train(
        data=data_path,
        epochs=50,          # Number of training rounds
        imgsz=640,          # Image size
        
        # IMPORTANT: Reduced batch size to 8 to fit in 8GB VRAM
        # If this runs without error, you can try increasing it to 10 or 12.
        batch=8,           
        
        name='yolov8m_custom_training', # Updated name for the run folder
        device=0            # Forces training on your NVIDIA GPU
    )
    
    print("Training complete.")

if __name__ == '__main__':
    main()
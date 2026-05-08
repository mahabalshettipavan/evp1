from ultralytics import YOLO
import os

def main():
    # Load a model
    # 'yolov8s.pt' will be automatically downloaded if not present
    model = YOLO('yolov8s.pt')  

    # Define the path to the data.yaml file
    # Ensure this path is correct relative to where you run the script
    data_path = os.path.abspath(os.path.join('unified_dataset', 'data.yaml'))

    if not os.path.exists(data_path):
        print(f"Error: data.yaml not found at {data_path}")
        print("Please run merge_dataset.py first to generate the dataset.")
        return

    # Train the model
    print("Starting training...")
    results = model.train(
        data=data_path,
        epochs=50,          # You can adjust the number of epochs
        imgsz=640,          # Image size
        batch=16,           # Batch size
        name='yolov8s_optimized', # Name of the run
        patience=10,        # Stop if no improvement for 10 epochs (Early Stopping)
        dropout=0.2,        # Randomly drop 20% of connections (Regularization)
        
        # Augmentation to generalize better
        degrees=10.0,       # +/- 10 degrees rotation
        shear=2.0,          # +/- 2 degrees shear
        perspective=0.0005, # Slight perspective change
        mixup=0.1,          # Mix 10% of images with others
        copy_paste=0.1      # Copy-paste augmentation
    )
    
    print("Training complete.")

if __name__ == '__main__':
    main()
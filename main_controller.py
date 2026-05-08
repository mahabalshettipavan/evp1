import traci
import time
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
SUMO_CMD = ["sumo-gui", "-c", "sumo_config/simulation.sumocfg"]
# CHANGE THIS to your actual Traffic Light ID from NetEdit
TRAFFIC_LIGHT_ID = "J1"
# CHANGE THIS to the phase index that gives the Ambulance a GREEN light
GREEN_PHASE_INDEX = 0
MODEL_PATH = r"C:\Users\PAVAN\Desktop\MINIPROJECT\yolov8s.pt"

def run_simulation():
    # 1. Load the AI Model
    print("Loading AI Model...")
    model = YOLO(MODEL_PATH)

    # 2. Start SUMO
    print("Starting Simulation...")
    traci.start(SUMO_CMD)
    
    # 3. Setup the Camera View (Optional but recommended)
    # This zooms the GUI camera onto the intersection so YOLO can see better
    # 'View#0' is the default view ID in SUMO
    traci.gui.setZoom("View#0", 1000) 
    
    step = 0
    emergency_mode = False

    while traci.simulation.getMinExpectedNumber() > 0:
        # A. Advance Simulation
        traci.simulationStep()
        
        # B. Capture "Camera" Frame
        # We save the current view to a temporary image file
        img_path = "temp_view.jpg"
        traci.gui.screenshot("View#0", img_path)
        
        # We need to wait a tiny bit for the file to actually save to disk
        # (In a real optimized system we would use memory buffers, but this is safer for now)
        time.sleep(0.05) 

        # C. AI Detection (Only if image exists)
        if os.path.exists(img_path):
            results = model(img_path, verbose=False) # verbose=False keeps terminal clean
            
            ambulance_detected = False
            
            # Check detection results
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    conf = float(box.conf)
                    
                    # Check if it sees an ambulance with high confidence
                    if class_name == "ambulance" and conf > 0.5:
                        ambulance_detected = True
                        print(f"🚨 AMBULANCE DETECTED at Step {step}! Confidence: {conf:.2f}")
            
            # D. Traffic Control Logic
            if ambulance_detected:
                if not emergency_mode:
                    print("--> FORCE GREEN LIGHT")
                    # Force the light to the Green Phase
                    traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, GREEN_PHASE_INDEX)
                    # Freeze the phase (prevent it from changing automatically)
                    traci.trafficlight.setPhaseDuration(TRAFFIC_LIGHT_ID, 10) 
                    emergency_mode = True
            else:
                # If no ambulance, logic to return to normal (optional for now)
                # For this prototype, we just let the duration expire
                pass
                
        step += 1

    traci.close()
    print("Simulation Finished.")

if __name__ == "__main__":
    run_simulation()
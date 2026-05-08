import os
import sys
import traci

# Setup SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

def run():
    print("Simulation started. Look for the Ambulance at step 50!")
    step = 0
    
    # Run until no vehicles are left
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        # Get list of all vehicle IDs currently on the road
        vehicle_ids = traci.vehicle.getIDList()
        
        for v_id in vehicle_ids:
            # Check vehicle type
            v_type = traci.vehicle.getTypeID(v_id)
            
            if v_type == "ambulance":
                # Get the lane the ambulance is currently in
                lane_id = traci.vehicle.getLaneID(v_id)
                # Get the traffic light ahead (if any)
                # This logic checks if the ambulance is approaching the center
                print(f"Time {step}: Ambulance {v_id} is on lane {lane_id}")

        step += 1
    
    traci.close()
    print("Simulation finished.")

if __name__ == "__main__":
    # We use sumo-gui so you can see the cars and ambulance
    sumoBinary = "sumo-gui"
    
    # Load the config we generated in Step 1
    sumoCmd = [sumoBinary, "-c", "simulation.sumocfg", "--start"]
    
    traci.start(sumoCmd)
    run()
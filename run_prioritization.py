import os
import sys
import traci

# --- Check for SUMO_HOME ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

def run():
    print("starting simulation with PRIORITY LOGIC...")
    step = 0
    
    # The ID of our traffic light (defined in nodes.nod.xml)
    TLS_ID = "center"
    
    # Define which phases correspond to which directions
    # Note: These indices (0 and 2) are standard SUMO defaults for a 4-way intersection.
    # Phase 0: "GGGrrrGGGrrr" (North-South Green)
    # Phase 2: "rrrGGGrrrGGG" (East-West Green)
    PHASE_NS_GREEN = 0
    PHASE_EW_GREEN = 2

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        # 1. DETECT: Scan for Emergency Vehicles
        ambulance_detected = False
        ambulance_edge = ""
        
        vehicle_ids = traci.vehicle.getIDList()
        for v_id in vehicle_ids:
            v_type = traci.vehicle.getTypeID(v_id)
            
            # We look for ambulances or firetrucks
            if v_type in ["ambulance", "firetruck"]:
                # Get the edge (road) id the vehicle is currently on
                edge_id = traci.vehicle.getRoadID(v_id)
                
                # Check if it is on an INCOMING lane (approaching the light)
                # Our incoming edges are N2C, S2C, E2C, W2C
                if edge_id in ["N2C", "S2C", "E2C", "W2C"]:
                    ambulance_detected = True
                    ambulance_edge = edge_id
                    
                    # Calculate distance to traffic light (optional debug info)
                    lane_len = traci.lane.getLength(traci.vehicle.getLaneID(v_id))
                    pos = traci.vehicle.getLanePosition(v_id)
                    dist_to_light = lane_len - pos
                    
                    print(f"Step {step}: EMERGENCY VEHICLE approaching on {edge_id}. Dist: {dist_to_light:.1f}m")
                    break # Prioritize the first one found

        # 2. ACT: Control Traffic Lights based on Detection
        if ambulance_detected:
            current_phase = traci.trafficlight.getPhase(TLS_ID)
            
            # If Ambulance is coming from North or South -> Force NS Green
            if ambulance_edge in ["N2C", "S2C"]:
                if current_phase != PHASE_NS_GREEN:
                    traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)
                    # Reset the timer so it stays green
                    traci.trafficlight.setPhaseDuration(TLS_ID, 1000) 
            
            # If Ambulance is coming from East or West -> Force EW Green
            elif ambulance_edge in ["E2C", "W2C"]:
                if current_phase != PHASE_EW_GREEN:
                    traci.trafficlight.setPhase(TLS_ID, PHASE_EW_GREEN)
                    traci.trafficlight.setPhaseDuration(TLS_ID, 1000)

        step += 1
    
    traci.close()

if __name__ == "__main__":
    # Start SUMO with GUI
    sumoBinary = "sumo-gui"
    sumoCmd = [sumoBinary, "-c", "simulation.sumocfg", "--start"]
    
    traci.start(sumoCmd)
    run()  
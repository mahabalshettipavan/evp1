import os
import sys
import traci

# Check for SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

def run_simulation(enable_priority):
    """
    Runs the simulation and returns a dictionary of travel times for emergency vehicles.
    """
    print(f"--- Starting Simulation | Priority Mode: {enable_priority} ---")
    
    # Start SUMO
    # We use --no-step-log to keep the console clean
    sumoBinary = "sumo" # Use "sumo" (no GUI) for faster calculation, or "sumo-gui" to watch
    sumoCmd = [sumoBinary, "-c", "simulation.sumocfg", "--start", "--no-step-log"]
    traci.start(sumoCmd)

    # Data storage
    vehicle_start_times = {}
    travel_times = {}
    
    TLS_ID = "center"
    PHASE_NS_GREEN = 0
    PHASE_EW_GREEN = 2
    
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        # --- 1. DATA COLLECTION ---
        # Track when vehicles enter the simulation
        for v_id in traci.simulation.getDepartedIDList():
            if traci.vehicle.getTypeID(v_id) in ["ambulance", "firetruck"]:
                vehicle_start_times[v_id] = step
        
        # Track when vehicles finish (leave) the simulation
        for v_id in traci.simulation.getArrivedIDList():
            if v_id in vehicle_start_times:
                duration = step - vehicle_start_times[v_id]
                travel_times[v_id] = duration

        # --- 2. PRIORITY LOGIC (Only if enabled) ---
        if enable_priority:
            ambulance_detected = False
            ambulance_edge = ""
            
            # Scan vehicles
            vehicle_ids = traci.vehicle.getIDList()
            for v_id in vehicle_ids:
                if traci.vehicle.getTypeID(v_id) in ["ambulance", "firetruck"]:
                    edge_id = traci.vehicle.getRoadID(v_id)
                    if edge_id in ["N2C", "S2C", "E2C", "W2C"]:
                        ambulance_detected = True
                        ambulance_edge = edge_id
                        break 

            # Act on Traffic Lights
            if ambulance_detected:
                current_phase = traci.trafficlight.getPhase(TLS_ID)
                if ambulance_edge in ["N2C", "S2C"] and current_phase != PHASE_NS_GREEN:
                    traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)
                    traci.trafficlight.setPhaseDuration(TLS_ID, 1000)
                elif ambulance_edge in ["E2C", "W2C"] and current_phase != PHASE_EW_GREEN:
                    traci.trafficlight.setPhase(TLS_ID, PHASE_EW_GREEN)
                    traci.trafficlight.setPhaseDuration(TLS_ID, 1000)

        step += 1
        
    traci.close()
    return travel_times

if __name__ == "__main__":
    # 1. Run Baseline (No Logic)
    print("Running Baseline Simulation (Normal Traffic)...")
    results_baseline = run_simulation(enable_priority=False)
    
    # 2. Run Smart Simulation (With Priority)
    print("\nRunning Smart Simulation (With Prioritization)...")
    results_priority = run_simulation(enable_priority=True)
    
    # 3. Print Comparison Report
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS REPORT")
    print("="*50)
    print(f"{'Vehicle ID':<15} | {'Baseline Time':<15} | {'Priority Time':<15} | {'Time Saved':<15}")
    print("-" * 65)
    
    total_saved = 0
    
    for v_id in results_baseline:
        base_time = results_baseline.get(v_id, 0)
        prio_time = results_priority.get(v_id, 0)
        saved = base_time - prio_time
        total_saved += saved
        
        print(f"{v_id:<15} | {base_time:<15} | {prio_time:<15} | {saved:<15}")

    print("-" * 65)
    print(f"Total Time Saved: {total_saved} steps")
    print("="*50)
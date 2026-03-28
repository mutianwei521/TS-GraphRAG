"""
Utility functions for water distribution network hydraulic simulation.
"""
import numpy as np
import wntr


def run_hydraulic_simulation(wn):
    """
    Run hydraulic simulation and return results.
    
    Args:
        wn: WNTR water network model
        
    Returns:
        sim_results: Simulation results object
    """
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    return results


def calculate_average_pressure(results, wn):
    """
    Calculate average pressure for each node across all time steps.
    
    Args:
        results: WNTR simulation results
        wn: WNTR water network model
        
    Returns:
        avg_pressure: Dictionary mapping node names to average pressure values
    """
    # Get pressure data for all nodes across all time steps
    pressure_df = results.node['pressure']
    
    # Calculate average pressure across all time steps for each node
    avg_pressure = pressure_df.mean(axis=0)
    
    # Convert to dictionary for easier access
    avg_pressure_dict = avg_pressure.to_dict()
    
    return avg_pressure_dict


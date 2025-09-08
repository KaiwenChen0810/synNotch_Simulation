"""
Calibration Analysis for synNotch Circuit Simulation
===================================================

This script calculates the correspondence between simulation parameters and real-world values
based on L929 cell characteristics and migration speeds.
"""

import numpy as np
import math
import os

def calculate_spatial_scale():
    """
    Calculate spatial scale: pixels to micrometers conversion
    Based on L929 cell size and target volume
    """
    # Real L929 cell parameters
    real_cell_diameter_um = 5  # micrometers (standard L929 cell size)
    real_cell_area_um2 = math.pi * (real_cell_diameter_um / 2) ** 2
    
    # Simulation parameters (from XML)
    target_volume_pixels = 25  # pixels^2 (2D simulation)
    
    # Calculate pixel to micrometer conversion
    # Assuming circular cells: area_pixels = area_um2 / (um_per_pixel)^2
    um_per_pixel = math.sqrt(real_cell_area_um2 / target_volume_pixels)
    
    return um_per_pixel

def calculate_field_of_view(um_per_pixel):
    """
    Calculate field of view in micrometers
    """
    grid_size_x = 120  # pixels
    grid_size_y = 120  # pixels
    
    fov_x_um = grid_size_x * um_per_pixel
    fov_y_um = grid_size_y * um_per_pixel
    
    return fov_x_um, fov_y_um

def load_migration_results():
    """
    Load migration speed results from migration test
    """
    if os.path.exists('migration_results.txt'):
        with open('migration_results.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('Average migration speed:'):
                    sim_speed_pixels_per_mcs = float(line.split(': ')[1].split(' ')[0])
                    return sim_speed_pixels_per_mcs
    return None

def calculate_temporal_scale(um_per_pixel, sim_speed_pixels_per_mcs):
    """
    Calculate temporal scale: MCS to minutes conversion
    Based on real L929 cell migration speed
    """
    # Real L929 cell migration speed
    real_speed_um_per_min = 0.488  # micrometers per minute
    
    # Convert simulation speed to um/MCS
    sim_speed_um_per_mcs = sim_speed_pixels_per_mcs * um_per_pixel
    
    # Calculate time scale conversion
    # real_speed_um_per_min = sim_speed_um_per_mcs / (min_per_mcs)
    min_per_mcs = sim_speed_um_per_mcs / real_speed_um_per_min
    
    return min_per_mcs

def print_complete_calibration():
    """
    Print complete calibration results
    """
    um_per_pixel = calculate_spatial_scale()
    fov_x, fov_y = calculate_field_of_view(um_per_pixel)
    
    print("=" * 70)
    print("COMPLETE SIMULATION CALIBRATION")
    print("=" * 70)
    
    # Spatial calibration
    print("\n📏 SPATIAL SCALE:")
    print(f"  Real L929 cell diameter: 5.0 μm")
    print(f"  Target volume in simulation: 25 pixels²")
    print(f"  Spatial scale: {um_per_pixel:.2f} μm/pixel")
    print(f"  Field of view: {fov_x:.1f} × {fov_y:.1f} μm")
    print(f"  Field of view: {fov_x/1000:.2f} × {fov_y/1000:.2f} mm")
    
    # Temporal calibration
    sim_speed = load_migration_results()
    if sim_speed:
        min_per_mcs = calculate_temporal_scale(um_per_pixel, sim_speed)
        
        print(f"\n⏱️  TEMPORAL SCALE:")
        print(f"  Real L929 migration speed: 0.488 μm/min")
        print(f"  Simulation migration speed: {sim_speed:.4f} pixels/MCS")
        print(f"  Simulation migration speed: {sim_speed * um_per_pixel:.4f} μm/MCS")
        print(f"  Temporal scale: {min_per_mcs:.2f} min/MCS")
        print(f"  Temporal scale: {min_per_mcs * 60:.1f} sec/MCS")
        
        
        
    else:
        print(f"\n❌ TEMPORAL CALIBRATION:")
        print(f"  Migration results not found. Run migration simulation first.")
    
    print("\n" + "=" * 70)
    return um_per_pixel, min_per_mcs if sim_speed else None

if __name__ == "__main__":
    # Run complete calibration
    um_per_pixel, min_per_mcs = print_complete_calibration()
    
    if min_per_mcs:
        print(f"\n✅ Calibration complete!")
        print(f"   Spatial: {um_per_pixel:.2f} μm/pixel")
        print(f"   Temporal: {min_per_mcs:.2f} min/MCS")
    else:
        print(f"\n⚠️  Run migration simulation to complete temporal calibration") 

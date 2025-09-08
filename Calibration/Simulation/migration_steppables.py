"""
Migration Test Steppables
========================

Tracks cell center of mass positions to calculate migration speeds
for temporal calibration of the simulation.
"""

from cc3d.core.PySteppables import SteppableBasePy
import numpy as np
import math

class MigrationTrackerSteppable(SteppableBasePy):
    """
    Tracks cell center of mass positions over time to calculate migration speeds
    """
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.cell_positions = {}  # Store position history for each cell
        self.cell_speeds = {}     # Store calculated speeds
        
    def start(self):
        """
        Initialize position tracking
        """
        # Record initial positions
        for cell in self.cell_list:
            if cell.type == 1:  # L929Cell
                self.cell_positions[cell.id] = [(0, cell.xCOM, cell.yCOM)]
                
    def step(self, mcs):
        """
        Track cell positions and calculate speeds
        """
        for cell in self.cell_list:
            if cell.type == 1:  # L929Cell
                cell_id = cell.id
                
                # Initialize if new cell
                if cell_id not in self.cell_positions:
                    self.cell_positions[cell_id] = []
                
                # Record current position
                self.cell_positions[cell_id].append((mcs, cell.xCOM, cell.yCOM))
                
                # Calculate instantaneous speed if we have enough history
                if len(self.cell_positions[cell_id]) >= 2:
                    recent_positions = self.cell_positions[cell_id][-2:]
                    mcs1, x1, y1 = recent_positions[0]
                    mcs2, x2, y2 = recent_positions[1]
                    
                    # Calculate distance and time
                    distance_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    time_mcs = mcs2 - mcs1
                    
                    if time_mcs > 0:
                        speed_pixels_per_mcs = distance_pixels / time_mcs
                        self.cell_speeds[cell_id] = speed_pixels_per_mcs
                        
    def finish(self):
        """
        Calculate and save migration statistics
        """
        # Calculate average migration speed
        if self.cell_speeds:
            speeds = list(self.cell_speeds.values())
            avg_speed = np.mean(speeds)
            std_speed = np.std(speeds)
            
            print(f"\n{'='*60}")
            print("MIGRATION SPEED ANALYSIS")
            print(f"{'='*60}")
            print(f"Number of cells tracked: {len(speeds)}")
            print(f"Average migration speed: {avg_speed:.4f} pixels/MCS")
            print(f"Standard deviation: {std_speed:.4f} pixels/MCS")
            print(f"Speed range: {min(speeds):.4f} - {max(speeds):.4f} pixels/MCS")
            
            # Save for temporal calibration
            with open('migration_results.txt', 'w') as f:
                f.write(f"Average migration speed: {avg_speed:.6f} pixels/MCS\n")
                f.write(f"Standard deviation: {std_speed:.6f} pixels/MCS\n")
                f.write(f"Number of cells: {len(speeds)}\n")
                
            print(f"Results saved to migration_results.txt")
            print(f"\nUse this data with calibration_analysis.py for temporal calibration")
        else:
            print("No migration data collected!")


class ResultsSteppable(SteppableBasePy):
    """
    Simple results saving for migration test
    """
    def __init__(self, frequency=100):
        SteppableBasePy.__init__(self, frequency)
        
    def step(self, mcs):
        """
        Print progress
        """
        if mcs % 500 == 0:
            num_cells = len([cell for cell in self.cell_list if cell.type == 1])
            print(f"MCS {mcs}: {num_cells} cells migrating") 

"""
synNotch Circuit Steppables
===========================

This file contains the steppable logic for the synNotch circuit simulation.
It implements:
1. Custom 5:1 A:B initialization (as per Science paper)
2. Surface-tethered CD19 tracking on A cells
3. Surface-tethered GFP tracking on B cells
4. Contact-dependent synNotch activation (both CD19->B and GFP->A)
5. E-cadherin modulation and reporter expression
6. No diffusion fields - all signaling is contact-dependent

CRITICAL: Both CD19 and GFP are surface-tethered, not diffusible
"""

from cc3d.core.PySteppables import SteppableBasePy
import numpy as np
from random import uniform, randint, seed, random


class RandomMixInitializerSteppable(SteppableBasePy):
    """
    Randomly converts 1/6 of A uninduced cells to B uninduced cells to achieve 5:1 ratio
    """
    def __init__(self, frequency=1, frac = 1.0/6.0):
        super().__init__(frequency)
        self.frac = frac
        
    def start(self):
        # Define cell types first
        self.AUNINDUCED = self.cell_type.AUninduced
        self.BUNINDUCED = self.cell_type.BUninduced
        
        # Count initial A cells
        initial_a_cells = len(self.cell_list_by_type(self.AUNINDUCED))
        print(f"Initial A cells: {initial_a_cells}")
        
        # For 5:1 ratio, we want final_A:final_B = 5:1
        # If we start with N A cells and convert X to B:
        # final_A = N - X, final_B = X
        # We want (N-X):X = 5:1, so N-X = 5X, therefore N = 6X, so X = N/6
        target_b_cells = initial_a_cells // 6  # Integer division for exact count
        print(f"Target B cells to create: {target_b_cells}")
        
        # Convert specific number rather than using random probability
        a_cells_list = list(self.cell_list_by_type(self.AUNINDUCED))
        np.random.seed(810)
        np.random.shuffle(a_cells_list)  # Randomize which cells get converted
        converted_count = 0
        for i in range(min(target_b_cells, len(a_cells_list))):
            a_cells_list[i].type = self.BUNINDUCED
            converted_count += 1
        
        # Report final counts
        final_a_cells = len(self.cell_list_by_type(self.AUNINDUCED))
        final_b_cells = len(self.cell_list_by_type(self.BUNINDUCED))
        
        print(f"After conversion: A cells = {final_a_cells}, B cells = {final_b_cells}")
        if final_b_cells > 0:
            ratio = final_a_cells / final_b_cells
            print(f"Ratio A:B = {final_a_cells}:{final_b_cells} = {ratio:.2f}:1")
        else:
            print("No B cells created")
        print(f"Converted {converted_count} cells from A to B")


class SurfaceCD19Tracker(SteppableBasePy):
    """
    Tracks surface-tethered CD19 expression on A cells
    """
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
    def start(self):
        """
        Initialize CD19 expression tracking
        """
        # Define cell types
        self.AUNINDUCED = self.cell_type.AUninduced
        self.AINDUCED = self.cell_type.AInduced
        self.BUNINDUCED = self.cell_type.BUninduced
        self.BINDUCED = self.cell_type.BInduced
        
    def step(self, mcs):
        """
        Track CD19 expression on A cells
        BIOLOGICAL: CD19 is constitutively expressed on ALL A cells
        This provides the initial signal to activate nearby B cells
        """
        if not hasattr(self, 'AUNINDUCED'):
            return
            
        for cell in self.cell_list:
            # A cells (both uninduced and induced) constitutively express surface-tethered CD19
            # This is the initial signal that can activate B cell synNotch pathway
            if cell.type == self.AUNINDUCED or cell.type == self.AINDUCED:
                cell.dict['surface_CD19'] = True
            else:
                cell.dict['surface_CD19'] = False


class SurfaceGFPTracker(SteppableBasePy):
    """
    Tracks gradual surface-tethered GFP expression on activated B cells
    B cells express GFP at HIGH rate (strong promoter) for reciprocal activation
    """
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
        # GFP expression parameters for B cells (HIGH rate)
        self.gfp_expression_rate = 0.08  # Fast accumulation for strong promoter
        self.gfp_visualization_threshold = 3.0  # Level needed for green color visualization
        self.gfp_max_level = 12.0  # Maximum GFP level
        
    def start(self):
        """
        Initialize GFP expression tracking
        """
        # Define cell types
        self.AUNINDUCED = self.cell_type.AUninduced
        self.AINDUCED = self.cell_type.AInduced
        self.BUNINDUCED = self.cell_type.BUninduced
        self.BINDUCED = self.cell_type.BInduced
        
    def step(self, mcs):
        """
        Gradually accumulate GFP on BInduced cells at HIGH rate
        """
        if not hasattr(self, 'BINDUCED'):
            return
            
        for cell in self.cell_list:
            # Initialize GFP tracking for all cells
            if 'surface_GFP_level' not in cell.dict:
                cell.dict['surface_GFP_level'] = 0.0
                cell.dict['surface_GFP'] = False
            
            # Only BInduced cells gradually express surface-tethered GFP at HIGH rate
            # Condition: either already BInduced **or** still BUninduced but has started protein expression
            if (cell.type == self.BINDUCED) or (cell.type == self.BUNINDUCED and cell.dict.get('protein_expression_started', False)):
                # Gradually accumulate GFP
                current_level = cell.dict['surface_GFP_level']
                new_level = min(current_level + self.gfp_expression_rate, self.gfp_max_level)
                cell.dict['surface_GFP_level'] = new_level
                
                # Set surface_GFP flag when threshold is reached for A cell activation
                # (A cells can be activated by any GFP+ B cell, regardless of level)
                if new_level > 0.0:
                    cell.dict['surface_GFP'] = True
                else:
                    cell.dict['surface_GFP'] = False
            else:
                # Non-BInduced cells have no GFP
                cell.dict['surface_GFP_level'] = 0.0
                cell.dict['surface_GFP'] = False


class SurfaceMcherryTracker(SteppableBasePy):
    """
    Tracks gradual surface-tethered mCherry expression on activated A cells
    A cells express mCherry at VERY LOW rate (weak promoter) - takes much longer for mature phenotype
    """
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
        # mCherry expression parameters for A cells (VERY LOW rate)
        self.mcherry_expression_rate = 0.069  # Adjusted for 10-min visibility (10 min / 0.23 min/MCS = 43.5 MCS; 3.0/43.5 = 0.069)
        self.mcherry_visualization_threshold = 3.0  # Higher threshold for red visualization
        self.mcherry_mature_threshold = 3.0  # Level needed for mature AInduced phenotype
        self.mcherry_max_level = 8.0  # Maximum mCherry level
        
    def start(self):
        """
        Initialize mCherry expression tracking
        """
        # Define cell types
        self.AUNINDUCED = self.cell_type.AUninduced
        self.AINDUCED = self.cell_type.AInduced
        self.BUNINDUCED = self.cell_type.BUninduced
        self.BINDUCED = self.cell_type.BInduced
        
    def step(self, mcs):
        """
        Gradually accumulate mCherry on AInduced cells at VERY LOW rate
        """
        if not hasattr(self, 'AINDUCED'):
            return
            
        for cell in self.cell_list:
            # Initialize mCherry tracking for all cells
            if 'surface_mCherry_level' not in cell.dict:
                cell.dict['surface_mCherry_level'] = 0.0
                cell.dict['surface_mCherry'] = False
                cell.dict['mature_AInduced'] = False
            
            # Only AInduced cells gradually express surface-tethered mCherry at VERY LOW rate
            # Condition: either already AInduced **or** still AUninduced but has started protein expression
            if (cell.type == self.AINDUCED) or (cell.type == self.AUNINDUCED and cell.dict.get('protein_expression_started', False)):
                # Gradually accumulate mCherry (VERY LOW rate)
                current_level = cell.dict['surface_mCherry_level']
                new_level = min(current_level + self.mcherry_expression_rate, self.mcherry_max_level)
                cell.dict['surface_mCherry_level'] = new_level
                
                # Set surface_mCherry flag when visible level is reached (lower threshold)
                if new_level >= self.mcherry_visualization_threshold:  # Changed threshold
                    cell.dict['surface_mCherry'] = True
                else:
                    cell.dict['surface_mCherry'] = False
                
                # Set mature_AInduced flag when sufficient mCherry is accumulated
                if new_level >= self.mcherry_mature_threshold:
                    cell.dict['mature_AInduced'] = True
                else:
                    cell.dict['mature_AInduced'] = False
            else:
                # Non-AInduced cells have no mCherry
                cell.dict['surface_mCherry_level'] = 0.0
                cell.dict['surface_mCherry'] = False
                cell.dict['mature_AInduced'] = False


class SynNotchActivationSteppable(SteppableBasePy):
    """
    Handles contact-dependent synNotch activation with proper biological sequence
    Activation → Cell sorting → Protein accumulation → Color change
    """
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
        # Biological timing parameters
        self.synnotch_activation_delay = 50   # Delay for synNotch activation
        # protein expression starts after synnotch_activation_delay; no fixed delay for phenotype switch – it depends on protein threshold
        
        # Contact requirements for activation - MORE STRINGENT for A cell activation
        self.min_contact_area = 2  # Higher minimum contact area for activation
        self.min_contact_duration = 1000  #2500 # Longer cumulative contact for A cell activation (GFP pathway)
        self.min_contact_duration_b = 600  # 600Shorter cumulative contact for B cell activation (CD19 pathway)
        
        # Track activation timing and contact history
        self.cell_activation_times = {}
        self.cell_activation_signals = {}
        
        # Track cumulative contact with phenotype (not individual cells)
        self.cell_contact_with_phenotype = {}  # Track contact duration with specific phenotypes
        self.cell_last_contact_mcs = {}  # Track when contact was last made

        # NEW: cumulative contact tracker for both A-cell and B-cell activation with decay
        self.cell_cumulative_contact = {}  # {cell_id: {'GFP': value, 'CD19': value}}

        # Decay strategy parameters
        self.decay_mode =  'percentage'#"subtraction"  # "subtraction" or "percentage"
        
        # Penalty applied each MCS without contact
        # For subtraction mode: subtract this value each MCS
        # For percentage mode: multiply by this factor each MCS
        self.contact_decay_a = 1.4  # For A cell GFP contact (subtraction: subtract 1.4, percentage: multiply by 0.95)
        self.contact_decay_b = 0.7  # For B cell CD19 contact (subtraction: subtract 0.7, percentage: multiply by 0.95)
        
        # Percentage decay factors (used when decay_mode = "percentage")
        self.contact_decay_percentage_a = 0.9998  # Multiply by this when A cell loses GFP contact
        self.contact_decay_percentage_b = 0.9998  # Multiply by this when B cell loses CD19 contact
        
        # Grace period: only start decaying after N consecutive MCS without contact
        # This mimics NICD/GFP persistence in real cells (minutes to hours half-life)
        self.decay_grace_period = 10  # MCS of no contact before decay starts
        
        # Track consecutive MCS without contact for each cell
        self.cell_no_contact_streak = {}  # {cell_id: {'GFP': count, 'CD19': count}}

    def set_decay_mode(self, mode, decay_a=None, decay_b=None, grace_period=None):
        """
        Convenience method to set decay mode and parameters
        
        Parameters:
        mode: "subtraction" or "percentage"
        decay_a: decay parameter for A cell GFP contact
        decay_b: decay parameter for B cell CD19 contact
        grace_period: MCS of no contact before decay starts (default 5)
        """
        if mode not in ["subtraction", "percentage"]:
            raise ValueError("decay_mode must be 'subtraction' or 'percentage'")
        
        self.decay_mode = mode
        if decay_a is not None:
            if mode == "subtraction":
                self.contact_decay_a = decay_a
            else:  # percentage
                self.contact_decay_percentage_a = decay_a
        if decay_b is not None:
            if mode == "subtraction":
                self.contact_decay_b = decay_b
            else:  # percentage
                self.contact_decay_percentage_b = decay_b
        if grace_period is not None:
            self.decay_grace_period = grace_period
        
        print(f"Decay mode set to '{mode}' with parameters: A={decay_a}, B={decay_b}, grace_period={grace_period}")

    def start(self):
        """
        Initialize activation tracking
        """
        # Define cell types
        self.AUNINDUCED = self.cell_type.AUninduced
        self.AINDUCED = self.cell_type.AInduced
        self.BUNINDUCED = self.cell_type.BUninduced
        self.BINDUCED = self.cell_type.BInduced

    def _update_contact_with_phenotype(self, mcs):
        """
        Update cumulative contact duration with specific phenotypes
        """
        # Initialize contact tracking for new cells
        for cell in self.cell_list:
            cell_id = cell.id
            if cell_id not in self.cell_contact_with_phenotype:
                self.cell_contact_with_phenotype[cell_id] = {'CD19': 0, 'GFP': 0}
                self.cell_last_contact_mcs[cell_id] = {'CD19': 0, 'GFP': 0}
            if cell_id not in self.cell_cumulative_contact:
                self.cell_cumulative_contact[cell_id] = {'GFP': 0.0, 'CD19': 0.0}
            if cell_id not in self.cell_no_contact_streak:
                self.cell_no_contact_streak[cell_id] = {'GFP': 0, 'CD19': 0}
        
        # Update contact duration for all cells
        for cell in self.cell_list:
            cell_id = cell.id
            
            # Track if cell is currently in contact with each phenotype
            has_cd19_contact = False
            has_gfp_contact = False
            
            # Check all neighbors for contact
            for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cell):
                if not neighbor or common_surface_area < self.min_contact_area:
                    continue
                
                # Check for contact with CD19+ cells (for B cell activation)
                if neighbor.dict.get('surface_CD19', False):
                    has_cd19_contact = True
                
                # Check for contact with GFP+ cells (for A cell activation)
                if neighbor.dict.get('surface_GFP', False):
                    has_gfp_contact = True
            
            # Update CD19 contact duration (both consecutive and cumulative)
            if has_cd19_contact:
                if mcs - self.cell_last_contact_mcs[cell_id]['CD19'] <= 1:  # Continuous contact
                    self.cell_contact_with_phenotype[cell_id]['CD19'] += 1
                else:
                    self.cell_contact_with_phenotype[cell_id]['CD19'] = 1  # Reset if contact was lost
                self.cell_last_contact_mcs[cell_id]['CD19'] = mcs
                # Cumulative contact increases by 1 each MCS with contact
                self.cell_cumulative_contact[cell_id]['CD19'] += 1
                # Reset no-contact streak when in contact
                self.cell_no_contact_streak[cell_id]['CD19'] = 0
            else:
                # No contact this MCS - reset consecutive contact if contact was lost
                if mcs - self.cell_last_contact_mcs[cell_id]['CD19'] > 1:
                    self.cell_contact_with_phenotype[cell_id]['CD19'] = 0
                
                # Increment no-contact streak
                self.cell_no_contact_streak[cell_id]['CD19'] += 1
                
                # Apply decay only after grace period
                if self.cell_no_contact_streak[cell_id]['CD19'] >= self.decay_grace_period:
                    current_cum = self.cell_cumulative_contact[cell_id]['CD19']
                    if self.decay_mode == "subtraction":
                        current_cum = max(0.0, current_cum - self.contact_decay_b)
                    elif self.decay_mode == "percentage":
                        current_cum = current_cum * self.contact_decay_percentage_b
                    self.cell_cumulative_contact[cell_id]['CD19'] = current_cum
            
            # Update GFP contact duration
            if has_gfp_contact:
                if mcs - self.cell_last_contact_mcs[cell_id]['GFP'] <= 1:  # Continuous contact
                    self.cell_contact_with_phenotype[cell_id]['GFP'] += 1
                else:
                    self.cell_contact_with_phenotype[cell_id]['GFP'] = 1  # Reset if contact was lost
                self.cell_last_contact_mcs[cell_id]['GFP'] = mcs

                # Cumulative contact increases by 1 each MCS with contact
                self.cell_cumulative_contact[cell_id]['GFP'] += 1
                # Reset no-contact streak when in contact
                self.cell_no_contact_streak[cell_id]['GFP'] = 0
            else:
                # No contact this MCS - reset if contact was lost
                if mcs - self.cell_last_contact_mcs[cell_id]['GFP'] > 1:
                    self.cell_contact_with_phenotype[cell_id]['GFP'] = 0

                # Increment no-contact streak
                self.cell_no_contact_streak[cell_id]['GFP'] += 1
                
                # Apply decay only after grace period
                if self.cell_no_contact_streak[cell_id]['GFP'] >= self.decay_grace_period:
                    current_cum = self.cell_cumulative_contact[cell_id]['GFP']
                    if self.decay_mode == "subtraction":
                        current_cum = max(0.0, current_cum - self.contact_decay_a)
                    elif self.decay_mode == "percentage":
                        current_cum = current_cum * self.contact_decay_percentage_a
                    self.cell_cumulative_contact[cell_id]['GFP'] = current_cum

    def step(self, mcs):
        """
        Contact-dependent synNotch activation with proper biological sequence
        """
        if not hasattr(self, 'AUNINDUCED'):
            return
            
        # Update contact history with phenotypes
        self._update_contact_with_phenotype(mcs)
        
        # Stage 1: Check for contact-dependent signal detection
        for cell in self.cell_list:
            cell_id = cell.id
            
            # Skip if already activated
            if cell_id in self.cell_activation_times:
                continue
                
            # CD19 signaling: B cells respond to cumulative contact with CD19+ A cells
            if cell.type == self.BUNINDUCED:
                cum_contact_cd19 = self.cell_cumulative_contact[cell_id]['CD19']
                if cum_contact_cd19 >= self.min_contact_duration_b:
                    self.cell_activation_times[cell_id] = mcs
                    self.cell_activation_signals[cell_id] = 'CD19'
                    cell.dict['activated_by_CD19'] = True  # Set activation flag
                    # print(f"MCS {mcs}: B cell {cell_id} activated by CD19 (cumulative contact: {cum_contact_cd19:.1f})")
            
            # GFP signaling: A cells respond to contact with GFP+ B cells
            elif cell.type == self.AUNINDUCED:
                cum_contact = self.cell_cumulative_contact[cell_id]['GFP']
                if cum_contact >= self.min_contact_duration:
                    self.cell_activation_times[cell_id] = mcs
                    self.cell_activation_signals[cell_id] = 'GFP'
                    cell.dict['activated_by_GFP'] = True  # Set activation flag
                    # print(f"MCS {mcs}: A cell {cell_id} activated by GFP (cumulative contact: {cum_contact:.1f})")
        
        # Stage 2: Start protein expression after synNotch activation delay
        for cell in self.cell_list:
            cell_id = cell.id
            if cell_id in self.cell_activation_times and not cell.dict.get('protein_expression_started', False):
                activation_time = self.cell_activation_times[cell_id]
                if mcs - activation_time >= self.synnotch_activation_delay:
                    cell.dict['protein_expression_started'] = True
                    # print(f"MCS {mcs}: Cell {cell_id} started protein expression")

        # Stage 3: Monitor protein accumulation; switch phenotype when threshold reached
        for cell in self.cell_list:
            # B cells becoming BInduced
            if cell.type == self.BUNINDUCED and cell.dict.get('protein_expression_started', False):
                if cell.dict.get('surface_GFP_level', 0.0) >= 3.0:  # same threshold as visualization
                    cell.type = self.BINDUCED
                    # print(f"MCS {mcs}: B cell {cell.id} -> BInduced (GFP level {cell.dict['surface_GFP_level']:.2f})")
                    # clean up activation tracking
                    cell_id = cell.id
                    if cell_id in self.cell_activation_times:
                        del self.cell_activation_times[cell_id]
                        del self.cell_activation_signals[cell_id]

            # A cells becoming AInduced
            if cell.type == self.AUNINDUCED and cell.dict.get('protein_expression_started', False):
                if cell.dict.get('surface_mCherry_level', 0.0) >= 6.0:
                    cell.type = self.AINDUCED
                    # print(f"MCS {mcs}: A cell {cell.id} -> AInduced (mCherry level {cell.dict['surface_mCherry_level']:.2f})")
                    cell_id = cell.id
                    if cell_id in self.cell_activation_times:
                        del self.cell_activation_times[cell_id]
                        del self.cell_activation_signals[cell_id]


class CellSortingSteppable(SteppableBasePy):
    """
    Monitors cell sorting progress
    """
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        
    def start(self):
        """
        Initialize sorting monitoring
        """
        pass
        
    def step(self, mcs):
        """
        Monitor cell sorting progress
        """
        # Count cell types
        type_counts = {}
        for cell in self.cell_list:
            cell_type = cell.type
            type_counts[cell_type] = type_counts.get(cell_type, 0) + 1
        
        # if mcs % 100 == 0:
        #     print(f"MCS {mcs}: Cell counts - {type_counts}")



class VisualizationSteppable(SteppableBasePy):
    """
    Provides visualization feedback with proper biological sequence
    Activation → Cell sorting → Protein accumulation → Color change
    """
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
    def start(self):
        """
        Initialize visualization
        """
        # Define cell types
        self.AUNINDUCED = self.cell_type.AUninduced
        self.AINDUCED = self.cell_type.AInduced
        self.BUNINDUCED = self.cell_type.BUninduced
        self.BINDUCED = self.cell_type.BInduced
        
    def step(self, mcs):
        """
        Update visualization based on cell states and protein levels
        """
        if not hasattr(self, 'BINDUCED'):
            return
            
        # Visual feedback for key transitions and protein accumulation
        if mcs % 1000 == 0:
            binduced_count = sum(1 for cell in self.cell_list if cell.type == self.BINDUCED)
            ainduced_count = sum(1 for cell in self.cell_list if cell.type == self.AINDUCED)
            mature_ainduced_count = sum(1 for cell in self.cell_list if cell.dict.get('mature_AInduced', False))
            
            # Count activated but not yet visible cells
            activated_b_count = sum(1 for cell in self.cell_list if cell.dict.get('activated_by_CD19', False))
            activated_a_count = sum(1 for cell in self.cell_list if cell.dict.get('activated_by_GFP', False))
            
            # Calculate average protein levels
            gfp_levels = [cell.dict.get('surface_GFP_level', 0.0) for cell in self.cell_list if cell.type == self.BINDUCED]
            mcherry_levels = [cell.dict.get('surface_mCherry_level', 0.0) for cell in self.cell_list if cell.type == self.AINDUCED]
            
            avg_gfp = sum(gfp_levels) / len(gfp_levels) if gfp_levels else 0.0
            avg_mcherry = sum(mcherry_levels) / len(mcherry_levels) if mcherry_levels else 0.0
            
            # if activated_b_count > 0:
            #     print(f"MCS {mcs}: {activated_b_count} B cells activated (cell sorting started)")
            # if binduced_count > 0:
            #     print(f"MCS {mcs}: {binduced_count} BInduced cells (green, high E-cad, avg GFP: {avg_gfp:.2f})")
            # if activated_a_count > 0:
            #     print(f"MCS {mcs}: {activated_a_count} A cells activated (cell sorting started)")
            # if ainduced_count > 0:
            #     print(f"MCS {mcs}: {ainduced_count} AInduced cells (red, low E-cad, avg mCherry: {avg_mcherry:.2f})")
            # if mature_ainduced_count > 0:
            #     print(f"MCS {mcs}: {mature_ainduced_count} MATURE AInduced cells (fully red phenotype)")


class DataCollectionSteppable(SteppableBasePy):
    """
    Collects simulation data for analysis including activation states and mature phenotypes
    """
    def __init__(self, frequency=50):
        SteppableBasePy.__init__(self, frequency)
        
    def start(self):
        """
        Initialize data collection
        """
        # Define cell types
        self.AUNINDUCED = self.cell_type.AUninduced
        self.AINDUCED = self.cell_type.AInduced
        self.BUNINDUCED = self.cell_type.BUninduced
        self.BINDUCED = self.cell_type.BInduced
        
        # Create data file
        self.data_file = open('synNotch_data.txt', 'w')
        # Updated header for proper biological sequence tracking
        self.data_file.write('MCS,AUninduced,BUninduced,AInduced,BInduced,activated_B_cells,activated_A_cells,mature_AInduced,CD19_positive_cells,GFP_positive_cells,mCherry_positive_cells,avg_GFP_level,avg_mCherry_level,Total_cells\n')
        
        # Create separate cell count tracking file
        self.cell_count_file = open('total_cell_count.txt', 'w')
        self.cell_count_file.write('MCS\ttotal_cells\n')
        
    def step(self, mcs):
        """
        Collect data at each timestep including activation states and protein levels
        """
        if not hasattr(self, 'AUNINDUCED'):
            return
            
        # Count cell types
        auninduced_count = sum(1 for cell in self.cell_list if cell.type == self.AUNINDUCED)
        buninduced_count = sum(1 for cell in self.cell_list if cell.type == self.BUNINDUCED)
        ainduced_count = sum(1 for cell in self.cell_list if cell.type == self.AINDUCED)
        binduced_count = sum(1 for cell in self.cell_list if cell.type == self.BINDUCED)
        mature_ainduced_count = sum(1 for cell in self.cell_list if cell.dict.get('mature_AInduced', False))
        
        # Count activated cells (before phenotype change)
        activated_b_count = sum(1 for cell in self.cell_list if cell.dict.get('activated_by_CD19', False))
        activated_a_count = sum(1 for cell in self.cell_list if cell.dict.get('activated_by_GFP', False))
        
        # Count surface marker expression
        cd19_positive = sum(1 for cell in self.cell_list if cell.dict.get('surface_CD19', False))
        gfp_positive = sum(1 for cell in self.cell_list if cell.dict.get('surface_GFP', False))
        mcherry_positive = sum(1 for cell in self.cell_list if cell.dict.get('surface_mCherry', False))
        
        # Calculate average protein levels
        gfp_levels = [cell.dict.get('surface_GFP_level', 0.0) for cell in self.cell_list if cell.type == self.BINDUCED]
        mcherry_levels = [cell.dict.get('surface_mCherry_level', 0.0) for cell in self.cell_list if cell.type == self.AINDUCED]
        
        avg_gfp = sum(gfp_levels) / len(gfp_levels) if gfp_levels else 0.0
        avg_mcherry = sum(mcherry_levels) / len(mcherry_levels) if mcherry_levels else 0.0
        
        total_cells = auninduced_count + buninduced_count + ainduced_count + binduced_count
        
        # Write data
        self.data_file.write(f'{mcs},{auninduced_count},{buninduced_count},{ainduced_count},{binduced_count},{activated_b_count},{activated_a_count},{mature_ainduced_count},{cd19_positive},{gfp_positive},{mcherry_positive},{avg_gfp:.3f},{avg_mcherry:.3f},{total_cells}\n')
        self.data_file.flush()
        
        # Write cell count data every step (regardless of this steppable's frequency)
        self.cell_count_file.write(f'{mcs}\t{total_cells}\n')
        self.cell_count_file.flush()
        
        # Print update every 500 MCS for console monitoring
        if mcs % 500 == 0:
            print(f"[CellCountTracker] MCS={mcs}  total_cells={total_cells}")
        
    def finish(self):
        """
        Close data files
        """
        self.data_file.close()
        self.cell_count_file.close()
        print("Data collection complete") 

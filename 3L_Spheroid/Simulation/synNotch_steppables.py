from cc3d.core.PySteppables import SteppableBasePy
import numpy as np
from random import uniform, randint, seed, random
import os

try:
    from .results_path import get_results_dir
except ImportError:
    from results_path import get_results_dir


class RandomMixInitializerSteppable(SteppableBasePy):
    """
    Randomly converts 1/6 of A uninduced cells to B uninduced cells to achieve 5:1 ratio
    """
    def __init__(self, frequency=1, ratio = 5):
        super().__init__(frequency)
        self.ratio = ratio+1
        
    def start(self):
        # Define cell types first
        self.AUNINDUCED = self.cell_type.AUninduced
        self.BUNINDUCED = self.cell_type.BUninduced
        
        # Count initial A cells
        initial_a_cells = len(self.cell_list_by_type(self.AUNINDUCED))
        print(f"Initial A cells: {initial_a_cells}")
        
        # For 5:1 ratio, we want final_A:final_B = 5:1
        target_b_cells = int(initial_a_cells // self.ratio)  # Integer division for exact count
        print(f"Target B cells to create: {target_b_cells}")
        
        # Convert specific number rather than using random probability
        a_cells_list = list(self.cell_list_by_type(self.AUNINDUCED))
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
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

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
            
            # Only BInduced cells gradually express surface-tethered GFP
            if (cell.type == self.BINDUCED) or (cell.type == self.BUNINDUCED and cell.dict.get('protein_expression_started', False)):
                # Gradually accumulate GFP
                current_level = cell.dict['surface_GFP_level']
                new_level = min(current_level + self.gfp_expression_rate, self.gfp_max_level)
                cell.dict['surface_GFP_level'] = new_level
                
                # Set surface_GFP flag when threshold is reached for A cell activation
                if new_level > 0.0:
                    cell.dict['surface_GFP'] = True
                else:
                    cell.dict['surface_GFP'] = False
            else:
                # Non-BInduced cells have no GFP
                cell.dict['surface_GFP_level'] = 0.0
                cell.dict['surface_GFP'] = False


class SurfaceMcherryTracker(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
        # mCherry expression parameters for A cells
        self.mcherry_expression_rate = 0.069  
        self.mcherry_visualization_threshold = 3.0  # Threshold for red visualization
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
            
            # Only AInduced cells gradually express surface-tethered mCherry
            if (cell.type == self.AINDUCED) or (cell.type == self.AUNINDUCED and cell.dict.get('protein_expression_started', False)):
                # Gradually accumulate mCherry
                current_level = cell.dict['surface_mCherry_level']
                new_level = min(current_level + self.mcherry_expression_rate, self.mcherry_max_level)
                cell.dict['surface_mCherry_level'] = new_level
                
                # Set surface_mCherry flag when visible level is reached
                if new_level >= self.mcherry_visualization_threshold: 
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
                cell .dict['surface_mCherry_level'] = 0.0
                cell.dict['surface_mCherry'] = False
                cell.dict['mature_AInduced'] = False


class SynNotchActivationSteppable(SteppableBasePy):
    """
    Handles contact-dependent synNotch activation with proper biological sequence
    Activation → Cell sorting → Protein accumulation → Color change
    """
    def __init__(self, frequency=1, min_contact_duration=None, min_contact_duration_b=None):
        SteppableBasePy.__init__(self, frequency)
        
        # Biological timing parameters
        self.synnotch_activation_delay = 50   # Delay for synNotch activation
        
        # Contact requirements for activation - MORE STRINGENT for A cell activation
        self.min_contact_area = 1 

        # Allow overriding via kwargs or environment variables for parameter scans
        def _coerce_duration(value, default):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default

        env_min_a = os.environ.get("MIN_CONTACT_DURATION")
        env_min_b = os.environ.get("MIN_CONTACT_DURATION_B")

        self.min_contact_duration = _coerce_duration(
            min_contact_duration if min_contact_duration is not None else env_min_a,
            2200
        )  # Longer cumulative contact for A cell activation (GFP pathway)
        self.min_contact_duration_b = _coerce_duration(
            min_contact_duration_b if min_contact_duration_b is not None else env_min_b,
            600
        )  # Shorter cumulative contact for B cell activation (CD19 pathway)
        
        # Track activation timing and contact history
        self.cell_activation_times = {}
        self.cell_activation_signals = {}
        
        # Track cumulative contact with phenotype (not individual cells)
        self.cell_contact_with_phenotype = {}  # Track contact duration with specific phenotypes
        self.cell_last_contact_mcs = {}  # Track when contact was last made

        # Cumulative contact tracker for both A-cell and B-cell activation with decay
        self.cell_cumulative_contact = {}  # {cell_id: {'GFP': value, 'CD19': value}}

        # Decay strategy parameters
        self.decay_mode =  'percentage'  # "subtraction" or "percentage"
        
        # Penalty applied each MCS without contact
        # For subtraction mode: subtract this value each MCS
        # For percentage mode: multiply by this factor each MCS
        self.contact_decay_a = 1.4  # For A cell GFP contact (subtraction: subtract 1.4, percentage: multiply by 0.95)
        self.contact_decay_b = 0.7  # For B cell CD19 contact (subtraction: subtract 0.7, percentage: multiply by 0.95)
        
        # Percentage decay factors (used when decay_mode = "percentage")
        self.contact_decay_percentage_a = 0.9998  # Multiply by this when A cell loses GFP contact
        self.contact_decay_percentage_b = 0.9998  # Multiply by this when B cell loses CD19 contact
        
        # Grace period: only start decaying after N consecutive MCS without contact
        self.decay_grace_period = 10  # MCS of no contact before decay starts
        
        # Track consecutive MCS without contact for each cell
        self.cell_no_contact_streak = {}  # {cell_id: {'GFP': count, 'CD19': count}}

    def set_decay_mode(self, mode, decay_a=None, decay_b=None, grace_period=None):
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
    Collects simulation data for analysis including activation states and mature phenotypes.
    """

    def __init__(self, frequency=50):
        SteppableBasePy.__init__(self, frequency)
        self.results_dir = None
        self.data_file = None
        self.cell_count_file = None
        self.first_a_induction_mcs = None
        self.first_b_induction_mcs = None
        self.last_b_induction_mcs = None
        self.prev_binduced_count = 0
        self.final_binduced_clusters = 0

    def start(self):
        """
        Initialize data collection
        """
        # Define cell types
        self.AUNINDUCED = self.cell_type.AUninduced
        self.AINDUCED = self.cell_type.AInduced
        self.BUNINDUCED = self.cell_type.BUninduced
        self.BINDUCED = self.cell_type.BInduced

        # Ensure we have a shared results directory
        self.results_dir = get_results_dir()

        # Create data file inside the run-specific folder
        data_path = self.results_dir / "synNotch_data.txt"
        self.data_file = open(data_path, "w")
        header = (
            "MCS,AUninduced,BUninduced,AInduced,BInduced,"
            "activated_B_cells,activated_A_cells,mature_AInduced,"
            "CD19_positive_cells,GFP_positive_cells,mCherry_positive_cells,"
            "avg_GFP_level,avg_mCherry_level,Total_cells,"
            "BInduced_cluster_count,largest_BInduced_cluster\n"
        )
        self.data_file.write(header)

        # Create separate cell count tracking file in the same folder
        self.cell_count_file = open(self.results_dir / "total_cell_count.txt", "w")
        self.cell_count_file.write("MCS\ttotal_cells\n")

    def _compute_binduced_clusters(self):
        """
        Return (cluster_count, largest_cluster_size) for BInduced cells.
        """
        binduced_cells = list(self.cell_list_by_type(self.BINDUCED))
        visited = set()
        cluster_sizes = []

        for cell in binduced_cells:
            if cell.id in visited:
                continue

            stack = [cell]
            visited.add(cell.id)
            current_size = 0

            while stack:
                current = stack.pop()
                current_size += 1
                for neighbor, _ in self.get_cell_neighbor_data_list(current):
                    if neighbor and neighbor.type == self.BINDUCED and neighbor.id not in visited:
                        visited.add(neighbor.id)
                        stack.append(neighbor)

            cluster_sizes.append(current_size)

        if not cluster_sizes:
            return 0, 0

        return len(cluster_sizes), max(cluster_sizes)

    def _update_induction_timings(self, mcs, ainduced_count, binduced_count):
        if self.first_a_induction_mcs is None and ainduced_count > 0:
            self.first_a_induction_mcs = mcs
        if self.first_b_induction_mcs is None and binduced_count > 0:
            self.first_b_induction_mcs = mcs
        if binduced_count > self.prev_binduced_count:
            self.last_b_induction_mcs = mcs
        self.prev_binduced_count = binduced_count

    def step(self, mcs):
        """
        Collect data at each timestep including activation states and protein levels
        """
        if not hasattr(self, "AUNINDUCED"):
            return

        # Count cell types
        auninduced_count = sum(1 for cell in self.cell_list if cell.type == self.AUNINDUCED)
        buninduced_count = sum(1 for cell in self.cell_list if cell.type == self.BUNINDUCED)
        ainduced_count = sum(1 for cell in self.cell_list if cell.type == self.AINDUCED)
        binduced_count = sum(1 for cell in self.cell_list if cell.type == self.BINDUCED)
        mature_ainduced_count = sum(1 for cell in self.cell_list if cell.dict.get("mature_AInduced", False))

        # Count activated cells (before phenotype change)
        activated_b_count = sum(1 for cell in self.cell_list if cell.dict.get("activated_by_CD19", False))
        activated_a_count = sum(1 for cell in self.cell_list if cell.dict.get("activated_by_GFP", False))

        # Count surface marker expression
        cd19_positive = sum(1 for cell in self.cell_list if cell.dict.get("surface_CD19", False))
        gfp_positive = sum(1 for cell in self.cell_list if cell.dict.get("surface_GFP", False))
        mcherry_positive = sum(1 for cell in self.cell_list if cell.dict.get("surface_mCherry", False))

        # Calculate average protein levels
        gfp_levels = [cell.dict.get("surface_GFP_level", 0.0) for cell in self.cell_list if cell.type == self.BINDUCED]
        mcherry_levels = [cell.dict.get("surface_mCherry_level", 0.0) for cell in self.cell_list if cell.type == self.AINDUCED]

        avg_gfp = sum(gfp_levels) / len(gfp_levels) if gfp_levels else 0.0
        avg_mcherry = sum(mcherry_levels) / len(mcherry_levels) if mcherry_levels else 0.0

        total_cells = auninduced_count + buninduced_count + ainduced_count + binduced_count
        cluster_count, largest_cluster = self._compute_binduced_clusters()

        self._update_induction_timings(mcs, ainduced_count, binduced_count)

        # Write data
        self.data_file.write(
            f"{mcs},{auninduced_count},{buninduced_count},{ainduced_count},{binduced_count},"
            f"{activated_b_count},{activated_a_count},{mature_ainduced_count},{cd19_positive},"
            f"{gfp_positive},{mcherry_positive},{avg_gfp:.3f},{avg_mcherry:.3f},{total_cells},"
            f"{cluster_count},{largest_cluster}\n"
        )
        self.data_file.flush()

        # Write cell count data every step (regardless of this steppable's frequency)
        self.cell_count_file.write(f"{mcs}\t{total_cells}\n")
        self.cell_count_file.flush()

        # Print update every 500 MCS for console monitoring
        if mcs % 500 == 0:
            print(f"[CellCountTracker] MCS={mcs}  total_cells={total_cells}")

    def _write_summary(self):
        """
        Append one-time summary statistics to the data file.
        """
        summary_lines = ["\n# Summary Metrics\n"]
        summary_lines.append(f"# first_A_induction_step={self.first_a_induction_mcs if self.first_a_induction_mcs is not None else 'NA'}\n")
        summary_lines.append(f"# first_B_induction_step={self.first_b_induction_mcs if self.first_b_induction_mcs is not None else 'NA'}\n")
        summary_lines.append(f"# last_B_induction_step={self.last_b_induction_mcs if self.last_b_induction_mcs is not None else 'NA'}\n")

        if self.first_b_induction_mcs is not None and self.last_b_induction_mcs is not None:
            duration = self.last_b_induction_mcs - self.first_b_induction_mcs
        else:
            duration = "NA"
        summary_lines.append(f"# B_induction_duration={duration}\n")
        summary_lines.append(f"# final_BInduced_cluster_count={self.final_binduced_clusters}\n")

        self.data_file.writelines(summary_lines)

    def finish(self):
        """
        Close data files
        """
        if self.data_file:
            cluster_count, _ = self._compute_binduced_clusters()
            self.final_binduced_clusters = cluster_count
            self._write_summary()
            self.data_file.close()
        if self.cell_count_file:
            self.cell_count_file.close()
        print("Data collection complete")

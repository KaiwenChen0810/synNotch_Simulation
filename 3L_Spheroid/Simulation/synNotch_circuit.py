"""
synNotch Circuit Simulation Main File
"""

import cc3d.CompuCellSetup as CompuCellSetup

from .synNotch_steppables import (
    RandomMixInitializerSteppable,
    SurfaceCD19Tracker,
    SynNotchActivationSteppable,
    SurfaceGFPTracker,
    SurfaceMcherryTracker,
    CellSortingSteppable,
    VisualizationSteppable,
    DataCollectionSteppable
)


# Register all steppables (CustomInitializationSteppable first to setup 5:1 ratio)
CompuCellSetup.register_steppable(steppable=RandomMixInitializerSteppable(frequency=1))
CompuCellSetup.register_steppable(steppable=SurfaceCD19Tracker(frequency=1))
CompuCellSetup.register_steppable(steppable=SynNotchActivationSteppable(frequency=1))
CompuCellSetup.register_steppable(steppable=SurfaceGFPTracker(frequency=1))
CompuCellSetup.register_steppable(steppable=SurfaceMcherryTracker(frequency=1))
CompuCellSetup.register_steppable(steppable=CellSortingSteppable(frequency=10))
CompuCellSetup.register_steppable(steppable=VisualizationSteppable(frequency=1))
CompuCellSetup.register_steppable(steppable=DataCollectionSteppable(frequency=50))

CompuCellSetup.run() 

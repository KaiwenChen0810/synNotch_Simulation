"""
Migration Test Simulation
========================

Simple simulation to measure L929 cell migration speeds for temporal calibration.
"""

import cc3d.CompuCellSetup as CompuCellSetup
from migration_steppables import MigrationTrackerSteppable, ResultsSteppable

# Register steppables
CompuCellSetup.register_steppable(steppable=MigrationTrackerSteppable(frequency=10))
CompuCellSetup.register_steppable(steppable=ResultsSteppable(frequency=100))

CompuCellSetup.run() 

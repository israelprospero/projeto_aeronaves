# Sample script on how to use the performance function from designTool.
# Remember to save this script in the same directory as designTool.py

# IMPORTS
import designTool as dt
import numpy as np
import pprint

# Load a sample case already defined in designTools.py:
airplane = dt.standard_airplane('fokker100')

# Execute the geometry function
dt.geometry(airplane)

# Guess values for initial iteration
W0 = 467500.00000000000000
W_cruise = 446787.65092499996535
Mf_cruise = W_cruise / W0

# Execute the weight estimation
T0, T0vec, deltaS_wlan, CLmaxTO = dt.performance(W0, Mf_cruise, airplane)

# Print results
print("T0 = ",T0)
print("T0vec = ",T0vec)
print("deltaS_wlan = ",deltaS_wlan)
print("CLmaxTO = ",CLmaxTO)

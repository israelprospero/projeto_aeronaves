# Sample script on how to use the performance function from designTool.
# Remember to save this script in the same directory as designTool.py

# IMPORTS
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import designTool as dt
import numpy as np
import pprint

# Load a sample case already defined in designTools.py:
airplane = dt.standard_airplane('fokker100')

# Execute the geometry function
dt.geometry(airplane)

# Guess values for initial iteration
W0_guess = 467500.00000000000000
W_cruise = 446787.65092499996535
Mf_cruise = W_cruise / W0_guess

# Execute the weight estimation
T0, T0vec, deltaS_wlan, CLmaxTO = dt.performance(W0_guess, Mf_cruise, airplane)

# Print results
print("T0 = ",T0)
print("T0vec = ",T0vec)
print("deltaS_wlan = ",deltaS_wlan)
print("CLmaxTO = ",CLmaxTO)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Sample script on how to use the fuel_weight function from designTool.
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
W0_guess = 467500.00000000000000
range_cruise = 2426120.00000000000000

# Execute the fuel weight estimation
W_fuel, W_cruise = dt.fuel_weight(W0_guess, airplane, range_cruise)

# Print results and updated dicionary
print("W_fuel = ",W_fuel)
print("W_cruise = ",W_cruise) # This functions uses the Mf_cruise variable as second output, so if you want to check the values change the output in the designTool.py file
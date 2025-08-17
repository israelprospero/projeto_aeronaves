# Sample script on how to use the thrust_matching function from designTool.
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
T0_guess = 140250.00000000000000

# Execute the weight estimation
dt.thrust_matching(W0_guess, T0_guess, airplane)

# Print results
print("airplane['CLmaxTO'] = ",airplane['CLmaxTO'])
print("airplane['T0'] = ",airplane['T0'])
print("airplane['T0vec'] = ",airplane['T0vec'])
print("airplane['W0'] = ",airplane['W0'])
print("airplane['W_empty'] = ",airplane['W_empty'])
print("airplane['W_fuel'] = ",airplane['W_fuel'])
print("airplane['deltaS_wlan'] = ",airplane['deltaS_wlan'])

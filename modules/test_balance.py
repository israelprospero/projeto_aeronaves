# Sample script on how to use the balance function from designTool.
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

# Execute the weight and thrust estimation
dt.thrust_matching(W0_guess, T0_guess, airplane)

# Execute the balance analysis
dt.balance(airplane)

# Print results
print("airplane['xcg_fwd'] = ",airplane['xcg_fwd'])
print("airplane['xcg_aft'] = ",airplane['xcg_aft'])
print("airplane['xnp'] = ",airplane['xnp'])
print("airplane['SM_fwd'] = ",airplane['SM_fwd'])
print("airplane['SM_aft'] = ",airplane['SM_aft'])
print("airplane['tank_excess'] = ",airplane['tank_excess'])
print("airplane['V_maxfuel'] = ",airplane['V_maxfuel'])
print("airplane['CLv'] = ",airplane['CLv'])


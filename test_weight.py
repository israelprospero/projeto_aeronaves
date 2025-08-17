import modules.designTool as dt
from modules.utils import print_fuel_table


# Load the standard airplane
airplane = dt.standard_airplane('fokker100')

# Modify parameters if needed
#airplane['AR_w'] = 8.43

# Execute the geometry function to update dimensions
dt.geometry(airplane)

# Optional: plot the 3D aircraft geometry
#dt.plot3d(airplane)

# Define initial weight and thrust guesses
W0_guess = 40000 * dt.gravity
T0_guess = 0.3 * W0_guess

# Run weight estimation module
W0, W_empty, W_fuel, Mf_cruise = dt.weight(W0_guess, T0_guess, airplane)

print_fuel_table(airplane)

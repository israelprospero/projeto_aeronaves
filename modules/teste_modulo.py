import designTool as dt
import numpy as np
import pprint

airplane = dt . standard_airplane ('fokker100')

dt.geometry(airplane)

# Guess values for initial iteration
W0_guess = 467500
T0_guess = 140250

# Execute the weight and thrust estimation
dt.thrust_matching (W0_guess,T0_guess,airplane)

# Execute the balance analysis
dt.balance(airplane)

# Execute the landing gear analysis
dt.landing_gear(airplane)

# Print results
print(" airplane['frac_nlg_fwd'] = " , airplane['frac_nlg_fwd'])
print(" airplane['frac_nlg_aft'] = " , airplane['frac_nlg_aft'])
print(" airplane['alpha_tipback'] = " , airplane['alpha_tipback'])
print(" airplane['alpha_tailstrike'] = " , airplane['alpha_tailstrike'])
print(" airplane['phi_overturn'] = " , airplane['phi_overturn'])
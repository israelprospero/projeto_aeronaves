import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import designTool as dt
import matplotlib . pyplot as plt
import numpy as np
import pprint
from modules import utils as m

## Landing Gear

# airplane = dt. standard_airplane ('my_airplane_1')
# dt.geometry(airplane)
# # dt.plot3d(airplane)

# W0_guess = 50150*dt.gravity
# T0_guess = 0.3*W0_guess 

# dt.thrust_matching(W0_guess, T0_guess, airplane)
# dt.balance(airplane)
# dt.landing_gear(airplane)

# print("airplane['frac_nlg_fwd'] =", airplane['frac_nlg_fwd'])
# print("airplane['frac_nlg_aft'] =", airplane['frac_nlg_aft'])
# print("airplane['alpha_tipback'] (deg) =", airplane['alpha_tipback']*180/np.pi)
# print("airplane['alpha_tailstrike'] (deg) =", airplane['alpha_tailstrike']*180/np.pi)
# print("airplane['phi_overturn'] (deg) =", airplane['phi_overturn']*180/np.pi)

## Full Analysis
airplane = dt. standard_airplane ('my_airplane_1')
#dt.geometry(airplane)
#dt.plot3d(airplane)

dt.analyze(airplane, print_log=True, plot=True)
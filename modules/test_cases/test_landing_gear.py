import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import designTool as dt
import numpy as np
import pprint

airplane = dt.standard_airplane('fokker100')

dt.geometry(airplane)

W0_guess = 467500.00000000000000
T0_guess = 140250.00000000000000

dt . thrust_matching ( W0_guess , T0_guess , airplane )
dt . balance ( airplane )
dt . landing_gear ( airplane )

print (" airplane [ ' frac_nlg_fwd '] = " , airplane ['frac_nlg_fwd'])
print (" airplane [ ' frac_nlg_aft '] = " , airplane ['frac_nlg_aft'])
print (" airplane [ ' alpha_tipback '] = " , airplane ['alpha_tipback'])
print (" airplane [ ' alpha_tailstrike '] = " , airplane ['alpha_tailstrike'])
print (" airplane [ ' phi_overturn '] = " , airplane ['phi_overturn'])
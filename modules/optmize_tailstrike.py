import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.optimize import minimize
from modules import designTool as dt
from modules import utils as m
import numpy as np

airplane = dt.standard_airplane('my_airplane_1')
dt.geometry(airplane)

# Parameters to optimize: x_tailstrike, z_tailstrike
def set_params(params):
    airplane['x_tailstrike'], airplane['z_tailstrike'] = params
    
    dt.geometry(airplane)
    W0_guess = 50150*dt.gravity
    T0_guess = 0.3*W0_guess 
    
    dt.thrust_matching(W0_guess, T0_guess, airplane)
    dt.balance(airplane)
    dt.landing_gear(airplane)

def objective(params):
    set_params(params)
    ratios = np.array([
        #airplane['frac_nlg_fwd'],
        #airplane['frac_nlg_aft'],
        airplane['alpha_tipback']*180/np.pi,
        airplane['alpha_tailstrike']*180/np.pi,
        airplane['phi_overturn']*180/np.pi
    ])
    targets = np.array([17, 10, 50])
    return np.sum((ratios - targets)**2)

x0 = [23.4, -1.54]
bounds = [
    (20, 25),   # x_tailstrike
    (-2, -1),   # z_tailstrike
]

# Run optimization
result = minimize(objective, x0, bounds=bounds, method="SLSQP")

print("Optimal parameters:")
print("x_tailstrike:", result.x[0])
print("z_tailstrike:", result.x[1])

set_params(result.x)
ratios = np.array([
        #airplane['frac_nlg_fwd'],
        #airplane['frac_nlg_aft'],
        airplane['alpha_tipback']*180/np.pi,
        airplane['alpha_tailstrike']*180/np.pi,
        airplane['phi_overturn']*180/np.pi
    ])
print("Ratios achieved:", ratios)


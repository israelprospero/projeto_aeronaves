import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.optimize import minimize
from modules import designTool as dt
from modules import utils as m
import numpy as np

airplane = dt.standard_airplane('my_airplane_1')
dt.geometry(airplane)

# Parameters to optimize: xr_w, x_mlg, x_tank_c_w, x_n, x_nlg, c_tank_c_w, S_w
def set_params(params):
    airplane['xr_w'], airplane['x_mlg'], airplane['x_tank_c_w'], airplane['x_n'], airplane['x_nlg'], airplane['c_tank_c_w'], airplane['S_w']  = params
    
    dt.geometry(airplane)
    W0_guess = 50150*dt.gravity
    T0_guess = 0.3*W0_guess 
    dt.thrust_matching(W0_guess, T0_guess, airplane)
    dt.balance(airplane)

def objective(params):
    set_params(params)
    ratios = np.array([
        (airplane['xcg_fwd'] - airplane['xm_w']) / airplane['cm_w'],
        (airplane['xcg_aft'] - airplane['xm_w']) / airplane['cm_w'],
        (airplane['xcg_mlg'] - airplane['xm_w']) / airplane['cm_w'],
        (airplane['xnp'] - airplane['xm_w']) / airplane['cm_w'],
        airplane['tank_excess']        
    ])
    targets = np.array([0.1, 0.35, 0.55, 0.45, 0.01])
    return np.sum((ratios - targets)**2)

x0 = [10.5, 16.0, 0.45, 12.0, 4, 0.4, 82]
bounds = [
    (10.0, 11.5),   # xr_w
    (12, 17),       # x_mlg
    (0.2, 0.3),     # x_tank_c_w
    (11.0, 13),     # x_n
    (3, 6),         # x_nlg
    (0.3, 0.5),     # c_tank_c_w
    (80, 90)        # S_w
]

# Run optimization
result = minimize(objective, x0, bounds=bounds, method="SLSQP")

print("Optimal parameters:")
print("xr_w:", result.x[0])
print("x_mlg:", result.x[1])
print("x_tank_c_w:", result.x[2])
print("x_n:", result.x[3])
print("x_nlg:", result.x[4])
print("c_tank_c_w:", result.x[5])
print("S_w:", result.x[6])

set_params(result.x)
ratios = np.array([
    (airplane['xcg_fwd'] - airplane['xm_w']) / airplane['cm_w'],
    (airplane['xcg_aft'] - airplane['xm_w']) / airplane['cm_w'],
    (airplane['xcg_mlg'] - airplane['xm_w']) / airplane['cm_w'],
    (airplane['xnp'] - airplane['xm_w']) / airplane['cm_w'],
    airplane['tank_excess']
])
print("Ratios achieved:", ratios)
# print(airplane['c_flap_c_wing'])
# print(airplane['c_ail_c_wing'])


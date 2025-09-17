'''Not used'''

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.optimize import minimize
from modules import designTool as dt
from modules import utils as m
import numpy as np
from pprint import pprint

airplane = dt.standard_airplane('my_airplane_1')
dt.geometry(airplane)

x0 = [airplane['S_w'], airplane['AR_w'], airplane['xr_w'], airplane['AR_v'], airplane['x_n'], airplane['y_n'], airplane['z_n'], airplane['L_n'], airplane['D_n'], airplane['x_nlg'], airplane['x_mlg'], airplane['y_mlg'], airplane['x_tailstrike'], airplane['z_tailstrike'], airplane['c_tank_c_w'], airplane['x_tank_c_w'], airplane['b_tank_b_w_start'], airplane['b_tank_b_w_end'], airplane['c_flap_c_wing'], airplane['b_flap_b_wing'], airplane['c_ail_c_wing'], airplane['b_ail_b_wing']]

def set_params(params):
    airplane['S_w'], airplane['AR_w'], airplane['xr_w'], airplane['AR_v'], airplane['x_n'], airplane['y_n'], airplane['z_n'], airplane['L_n'], airplane['D_n'], airplane['x_nlg'], airplane['x_mlg'], airplane['y_mlg'], airplane['x_tailstrike'], airplane['z_tailstrike'], airplane['c_tank_c_w'], airplane['x_tank_c_w'], airplane['b_tank_b_w_start'], airplane['b_tank_b_w_end'], airplane['c_flap_c_wing'], airplane['b_flap_b_wing'], airplane['c_ail_c_wing'], airplane['b_ail_b_wing'] = params
    
    dt.analyze(airplane, print_log=False, plot=False)

def objective(params):
    set_params(params)
    ratios = np.array([
        airplane['deltaS_wlan'],
        (airplane['xcg_fwd'] - airplane['xm_w']) / airplane['cm_w'],
        (airplane['xcg_aft'] - airplane['xm_w']) / airplane['cm_w'],
        (airplane['xcg_mlg'] - airplane['xm_w']) / airplane['cm_w'],
        (airplane['xnp'] - airplane['xm_w']) / airplane['cm_w'],
        airplane['SM_fwd'],
        airplane['SM_aft'],
        airplane['CLv'],
        airplane['frac_nlg_fwd'],
        airplane['frac_nlg_aft'],
        airplane['alpha_tipback']*180/np.pi,
        airplane['alpha_tailstrike']*180/np.pi,
        airplane['phi_overturn']*180/np.pi,
        airplane['tank_excess']
                 
    ])
    targets = np.array([0.01, 0.10, 0.35, 0.55, 0.45, 0.35, 0.1, 0.5, 0.1, 0.06, 16, 11, 50, 0.1])
    return np.sum((ratios - targets)**2)

bounds = [
    (82, 90),                   # S_w
    (9, 9.6),                   # AR_w
    (9.639, 11.781),            # xr_w
    (1.3, 1.42),                # AR_v
    (10, 15),                   # x_n
    (4.0, 5.511),               # y_n
    (-2.585, -2.115),           # z_n
    (4.419, 5.401),             # L_n
    (1.521, 1.859),             # D_n
    (3.699, 4.521),             # x_nlg
    (14.022, 17.138),           # x_mlg
    (2.223, 2.717),             # y_mlg
    (22.3, 24.081),             # x_tailstrike
    (-1.01, -0.84),             # z_tailstrike
    (0.45, 0.5),                # c_tank_c_w
    (0.15, 0.2),                # x_tank_c_w
    (0.0, 0.05),                # b_tank_b_w_start
    (0.855, 0.95),              # b_tank_b_w_end
    (0.2, 0.3),                 # c_flap_c_wing
    (0.54, 0.66),               # b_flap_b_wing
    (0.2, 0.3),                 # c_ail_c_wing
    (0.3, 0.4)                  # b_ail_b_wing
]

# Run optimization
result = minimize(objective, x0, bounds=bounds, method="SLSQP")

for i, val in enumerate(result.x):
    print(val)

set_params(result.x)
ratios = np.array([
    airplane['deltaS_wlan'],
    (airplane['xcg_fwd'] - airplane['xm_w']) / airplane['cm_w'],
    (airplane['xcg_aft'] - airplane['xm_w']) / airplane['cm_w'],
    (airplane['xcg_mlg'] - airplane['xm_w']) / airplane['cm_w'],
    (airplane['xnp'] - airplane['xm_w']) / airplane['cm_w'],
    airplane['SM_fwd'],
    airplane['SM_aft'],
    airplane['CLv'],
    airplane['frac_nlg_fwd'],
    airplane['frac_nlg_aft'],
    airplane['alpha_tipback']*180/np.pi,
    airplane['alpha_tailstrike']*180/np.pi,
    airplane['phi_overturn']*180/np.pi,
    airplane['tank_excess'] 
])


names = [
    'deltaS_wlan',
    'xcg_fwd',
    'xcg_aft',
    'xcg_mlg',
    'xnp',
    'SM_fwd',
    'SM_aft',
    'CLv',
    'frac_nlg_fwd',
    'frac_nlg_aft',
    'alpha_tipback',
    'alpha_tailstrike',
    'phi_overturn',
    'tank_excess'
]

print('\n ----- RESULTS ------ \n')

for name, val in zip(names, ratios):
    print(f"{name}: {val:.6f}")

# pprint({k: v for k, v in airplane.items()})
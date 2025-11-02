import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.optimize import minimize
import auxmod as am
import matplotlib.pyplot as plt
import modules.designTool as dt
import copy
from pprint import pprint


airplane_base = dt.standard_airplane('my_airplane_1')
xlist, flist, g_hist = [], [], []
h_list = []

def get_5perc_var(val):
    """Return ±5% variation range."""
    if val > 0:
        return val - 0.05 * val, val + 0.05 * val
    else:
        return val + 0.05 * val, val - 0.05 * val


airplane_ref = dt.standard_airplane('my_airplane_1')
VAR_DICT = {
    'S_w':              [80, 100],
    'AR_w':             [8, 12],
    'taper_w':          list(get_5perc_var(airplane_ref['taper_w'])),
    'sweep_w':          list(get_5perc_var(airplane_ref['sweep_w'])),
    'dihedral_w':       list(get_5perc_var(airplane_ref['dihedral_w'])),
    'xr_w':             list(get_5perc_var(airplane_ref['xr_w'])),
    'zr_w':             list(get_5perc_var(airplane_ref['zr_w'])),
    'AR_h':             list(get_5perc_var(airplane_ref['AR_h'])),
    'taper_h':          list(get_5perc_var(airplane_ref['taper_h'])),
    'sweep_h':          list(get_5perc_var(airplane_ref['sweep_h'])),
    'dihedral_h':       list(get_5perc_var(airplane_ref['dihedral_h'])),
    'zr_h':             list(get_5perc_var(airplane_ref['zr_h'])),
    'AR_v':             list(get_5perc_var(airplane_ref['AR_v'])),
    'taper_v':          list(get_5perc_var(airplane_ref['taper_v'])),
    'sweep_v':          list(get_5perc_var(airplane_ref['sweep_v'])),
    'zr_v':             list(get_5perc_var(airplane_ref['zr_v'])),
    'x_n':              list(get_5perc_var(airplane_ref['x_n'])),
    'y_n':              list(get_5perc_var(airplane_ref['y_n'])),
    'z_n':              list(get_5perc_var(airplane_ref['z_n'])),
    'L_n':              list(get_5perc_var(airplane_ref['L_n'])),
    'D_n':              list(get_5perc_var(airplane_ref['D_n'])),
    'x_nlg':            list(get_5perc_var(airplane_ref['x_nlg'])),
    'x_mlg':            list(get_5perc_var(airplane_ref['x_mlg'])),
    'y_mlg':            list(get_5perc_var(airplane_ref['y_mlg'])),
    'z_lg':             list(get_5perc_var(airplane_ref['z_lg'])),
    'x_tailstrike':     list(get_5perc_var(airplane_ref['x_tailstrike'])),
    'z_tailstrike':     list(get_5perc_var(airplane_ref['z_tailstrike'])),
    'c_tank_c_w':       list(get_5perc_var(airplane_ref['c_tank_c_w'])),
    'x_tank_c_w':       list(get_5perc_var(airplane_ref['x_tank_c_w'])),
    'b_tank_b_w_end':   list(get_5perc_var(airplane_ref['b_tank_b_w_end'])),
    'c_flap_c_wing':    list(get_5perc_var(airplane_ref['c_flap_c_wing'])),
    'b_flap_b_wing':    list(get_5perc_var(airplane_ref['b_flap_b_wing'])),
    'c_ail_c_wing':     list(get_5perc_var(airplane_ref['c_ail_c_wing'])),
    'b_ail_b_wing':     list(get_5perc_var(airplane_ref['b_ail_b_wing']))
}

VAR_NAMES = list(VAR_DICT.keys())

x_min = np.array([VAR_DICT[k][0] for k in VAR_NAMES])
x_max = np.array([VAR_DICT[k][1] for k in VAR_NAMES])

# i = 0
# for name in VAR_NAMES:
#     print(f'{name}: {x_min[i]} and {x_max[i]}')
#     i += 1

# input()


x0_physical = np.array([airplane_ref[k] for k in VAR_NAMES])

def normalize(x): return (x - x_min) / (x_max - x_min)

def denormalize(xn): return x_min + xn * (x_max - x_min)

x0_norm = normalize(x0_physical)
bounds_norm = [(0.0, 1.0) for _ in VAR_NAMES]

def update_airplane(airplane, x):
    for key, val in zip(VAR_NAMES, x):
        airplane[key] = val
    return airplane


def run_analysis(x):

    airplane = update_airplane(copy.deepcopy(airplane_base), x)
    dt.analyze(airplane, print_log=False, plot=False)

    # minimize MTOW
    f = airplane['W0']  

    g = [
        airplane['deltaS_wlan'],
        0.4 - airplane['SM_fwd'],
        airplane['SM_aft'] - 0.05,
        0.75 - airplane['CLv'],
        0.15 - airplane['frac_nlg_fwd'],
        airplane['frac_nlg_aft'] - 0.04,
        airplane['alpha_tipback'] * 180 / np.pi - 15,
        airplane['alpha_tailstrike'] * 180 / np.pi - 10,
        63 - airplane['phi_overturn'] * 180 / np.pi,
    ]
    
    h = [
        airplane['tank_excess']
    ]

    return f, g, h


def objfun(xn):
    x = denormalize(xn)
    f, g, h = run_analysis(x)
    xlist.append(x)
    flist.append(f)
    g_hist.append(g)
    h_list.append(h)
    return f


def ineqconfun(xn):
    x = denormalize(xn)
    _, g, _ = run_analysis(x)
    return g

def eqconfun(xn):
    x = denormalize(xn)
    _, _, h = run_analysis(x)
    return h

con_ineq = {'type': 'ineq', 'fun': ineqconfun}
con_eq   = {'type': 'eq', 'fun': eqconfun}
cons = [con_ineq, con_eq]

result = minimize(objfun, x0_norm, constraints=cons, bounds=bounds_norm, method='SLSQP')
xopt_norm = result.x
xopt = denormalize(xopt_norm)

# print(result)

airplane_opt = update_airplane(copy.deepcopy(airplane_base), xopt)
dt.analyze(airplane_opt, print_log=True, plot=False)
# pprint(airplane_opt)

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

VAR_NAMES = [
    'S_w', 'AR_w', 'taper_w', 'sweep_w', 'dihedral_w', 'xr_w', 'zr_w',
    'AR_h', 'taper_h', 'sweep_h', 'dihedral_h', 'zr_h',
    'AR_v', 'taper_v', 'sweep_v', 'zr_v',
    'x_n', 'y_n', 'z_n', 'L_n', 'D_n',
    'x_nlg', 'x_mlg', 'y_mlg', 'z_lg',
    'x_tailstrike', 'z_tailstrike',
    'c_tank_c_w', 'x_tank_c_w', 'b_tank_b_w_start', 'b_tank_b_w_end',
    'c_flap_c_wing', 'b_flap_b_wing', 'c_ail_c_wing', 'b_ail_b_wing'
]

# Other candidates as inputs
# 'tcr_w' # t/c of the root section of the wing
# 'tct_w' # t/c of the tip section of the wing
# 'Cht'  # Horizontal tail volume coefficient
# 'Lc_h' # Non-dimensional lever of the horizontal tail (lever/wing_mac)
# 'tcr_h' # t/c of the root section of the HT
# 'tct_h' # t/c of the tip section of the HT
# 'Cvt' # Vertical tail volume coefficient
# 'Lb_v'  # Non-dimensional lever of the vertical tail (lever/wing_span)
# 'tcr_v' # t/c of the root section of the VT
# 'tct_v' # t/c of the tip section of the VT
# 'flap_type'  # Flap type
# 'slat_type' # Slat type
# 'c_slat_c_wing' # Fraction of the wing chord occupied by slats
# 'b_slat_b_wing' # Fraction of the wing span occupied by slats

xlist, flist = [], []
g_hist, h_hist = [], []

def update_airplane(airplane, x):

    for key, val in zip(VAR_NAMES, x):
        airplane[key] = val
    return airplane

def get_5perc_var(val):
    if val > 0:
        return val - 0.05*val, val + 0.05*val
    else:
        return val + 0.05*val, val - 0.05*val

def run_analysis(x):
    """
    Modifica uma cópia do dicionário airplane_base, executa dt.analyze(),
    e retorna objetivo e restrições.
    """
    airplane = update_airplane(copy.deepcopy(airplane_base), x)

    dt.analyze(airplane, print_log=False, plot=False)
    
    dt.analyze(airplane, print_log=False, plot=False)
    
    # Objetivo: minimizar o MTOW
    f = airplane['W0']

    # Restricoes de desigualdade (g(x) >= 0)
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
        airplane['tank_excess']
    ]            

    return f, g

def objfun(x):
    
    f, g = run_analysis(x)
    xlist.append(x)
    flist.append(f)
    g_hist.append(g)
    return f

def ineqconfun(x):
    _, g = run_analysis(x)
    return g

con_ineq = {'type': 'ineq', 'fun': ineqconfun}
cons = [con_ineq] #, con_eq]

airplane_ref = dt.standard_airplane('my_airplane_1')

# Bounds (variaveis de projeto)
bounds = [  [80, 100], # S_w
            [8,   12], # AR
            list(get_5perc_var(airplane_ref['taper_w'])), 
            list(get_5perc_var(airplane_ref['sweep_w'])), 
            list(get_5perc_var(airplane_ref['dihedral_w'])), 
            list(get_5perc_var(airplane_ref['xr_w'])), 
            list(get_5perc_var(airplane_ref['zr_w'])), 
            list(get_5perc_var(airplane_ref['AR_h'])), 
            list(get_5perc_var(airplane_ref['taper_h'])), 
            list(get_5perc_var(airplane_ref['sweep_h'])), 
            list(get_5perc_var(airplane_ref['dihedral_h'])), 
            list(get_5perc_var(airplane_ref['zr_h'])), 
            list(get_5perc_var(airplane_ref['AR_v'])), 
            list(get_5perc_var(airplane_ref['taper_v'])), 
            list(get_5perc_var(airplane_ref['sweep_v'])), 
            list(get_5perc_var(airplane_ref['zr_v'])), 
            list(get_5perc_var(airplane_ref['x_n'])), 
            list(get_5perc_var(airplane_ref['y_n'])), 
            list(get_5perc_var(airplane_ref['z_n'])), 
            list(get_5perc_var(airplane_ref['L_n'])), 
            list(get_5perc_var(airplane_ref['D_n'])), 
            list(get_5perc_var(airplane_ref['x_nlg'])), 
            list(get_5perc_var(airplane_ref['x_mlg'])), 
            list(get_5perc_var(airplane_ref['y_mlg'])), 
            list(get_5perc_var(airplane_ref['z_lg'])), 
            list(get_5perc_var(airplane_ref['x_tailstrike'])), 
            list(get_5perc_var(airplane_ref['z_tailstrike'])), 
            list(get_5perc_var(airplane_ref['c_tank_c_w'])), 
            list(get_5perc_var(airplane_ref['x_tank_c_w'])), 
            list(get_5perc_var(airplane_ref['b_tank_b_w_start'])), 
            list(get_5perc_var(airplane_ref['b_tank_b_w_end'])), 
            list(get_5perc_var(airplane_ref['c_flap_c_wing'])), 
            list(get_5perc_var(airplane_ref['b_flap_b_wing'])), 
            list(get_5perc_var(airplane_ref['c_ail_c_wing'])), 
            list(get_5perc_var(airplane_ref['b_ail_b_wing'])) ]

x0 = np.array([airplane_ref[v] for v in VAR_NAMES])

result = minimize(objfun, x0, constraints=cons, bounds=bounds, method='SLSQP')
xopt = result.x

print(result)

airplane_opt = update_airplane(copy.deepcopy(airplane_base), xopt)
dt.analyze(airplane_opt, print_log=True, plot=False)
# pprint(airplane_opt)

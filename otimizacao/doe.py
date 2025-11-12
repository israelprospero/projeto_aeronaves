import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from aux_tools_doe import corrdot
import auxmod as am
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt
import modules.designTool as dt
import copy
from pprint import pprint
import otimizacao.aerodinamica as aero
import otimizacao.pesos as pesos
import otimizacao.desempenho as desemp
import otimizacao.estabilidade as estab
import pprint

airplane_base = dt.standard_airplane('my_airplane_1')

def get_perc_var(val):
    perc = 0.1
    if val > 0:
        return val - perc * val, val + perc * val
    else:
        return val + perc * val, val - perc * val


airplane_ref = dt.standard_airplane('my_airplane_1')
VAR_DICT = {
    # 'var':            [lower bound, upper bound]
    'S_w':              [80, 100],
    'AR_w':             [7, 12],
    'taper_w':          [0.2, 0.4],
    'sweep_w':          [20*np.pi/180, 30*np.pi/180],
    'dihedral_w':       [2*np.pi/180, 5*np.pi/180],
    'xr_w':             list(get_perc_var(airplane_ref['xr_w'])),
    'zr_w':             list(get_perc_var(airplane_ref['zr_w'])),
    'tcr_w':            list(get_perc_var(airplane_ref['tcr_w'])),
    'tct_w':            list(get_perc_var(airplane_ref['tct_w'])),
    'Cht':              [0.9, 1.1],
    'Lc_h':             list(get_perc_var(airplane_ref['Lc_h'])),
    'AR_h':             [4, 5],
    'taper_h':          [0.3, 0.5],
    'sweep_h':          [20*np.pi/180, 35*np.pi/180],
    'dihedral_h':       [3*np.pi/180, 5*np.pi/180],
    'zr_h':             [0.6, 0.9],
    'tcr_h':            [0.05, 0.15],
    'tct_h':            [0.05, 0.15],
    'Cvt':              [0.06, 0.08],
    'Lb_v':             [0.35, 0.5],
    'AR_v':             [1, 2],
    'taper_v':          [0.3, 0.5],
    'sweep_v':          [30*np.pi/180, 50*np.pi/180],
    'zr_v':             [1, 3],
    'x_n':              [10, 18],
    'y_n':              [3, 6],
    'z_n':              [-4, -1],
    'L_n':              [3, 6],
    'D_n':              [1.3, 2],
    'x_nlg':            [3, 5],
    'x_mlg':            [13, 17],
    'y_mlg':            [2, 4],
    'z_lg':             [-4, -2],
    'x_tailstrike':     [20, 25],
    'z_tailstrike':     [-1.3, -0.8],
    'c_tank_c_w':       [0.45, 0.55],
    'b_tank_b_w_end':   [0.8, 0.95],
    'c_flap_c_wing':    [0.2, 0.3],
    'b_flap_b_wing':    [0.55, 0.7],
    'c_ail_c_wing':     [0.2, 0.35],
    'b_ail_b_wing':     [0.3, 0.4]
}

DOE_OUTPUT_NAMES = [
    'W0',
    'aero_CD_cruise',
    'aero_LD_cruise',
    'phi_overturn',
    'alpha_tailstrike',
    'alpha_tipback',
    'CLv',
    'tank_excess',
    'V_maxfuel',
    'SM_fwd',
    'SM_aft',
    'xnp',
    'deltaS_wlan',
    'T0',
    # 'CD0_altcruise',
    # 'CD0_cruise',
    'CLmaxTO',
    # 'CD',
    'aero_CLmax_landing'
]


VAR_NAMES = list(VAR_DICT.keys())

def update_airplane(airplane, x):
    for key, val in zip(VAR_NAMES, x):
        airplane[key] = val
    return airplane

def run_analysis(x):

    airplane = update_airplane(copy.deepcopy(airplane_base), x)
    airplane = dt.analyze(airplane, print_log=False, plot=False)
    
    airplane = aero.analise_aerodinamica(airplane, show_results=False)

    doe_out = {name: airplane[name] for name in DOE_OUTPUT_NAMES} 

    return doe_out

def analise_doe():
    n_samples = 100
    sampler = LHS()
    n_var = len(VAR_DICT)

    lb = []
    ub = []
    for key in VAR_NAMES:
        lb.append(VAR_DICT[key][0])
        ub.append(VAR_DICT[key][1])

    problem = Problem(n_var=n_var, xl=lb, xu=ub)
    samples = sampler(problem, n_samples).get("X")

    for i in range(n_var):
        samples[:,i] = lb[i] + (ub[i] - lb[i])*samples[:,i]
        
    doe_arrays = {name: np.zeros(n_samples) for name in DOE_OUTPUT_NAMES}
    for i in range(n_samples):
        doe_res = run_analysis(samples[i, :])
        for name in DOE_OUTPUT_NAMES:
            doe_arrays[name][i] = doe_res[name]
        
    data_dict = {var_name: samples[:, i] for i, var_name in enumerate(VAR_NAMES)}

    for name in DOE_OUTPUT_NAMES:
        data_dict[name] = doe_arrays[name]

    df = pd.DataFrame(data_dict)

    x_cols = VAR_NAMES
    y_cols = DOE_OUTPUT_NAMES
    correlation = df[x_cols + y_cols].corr().loc[x_cols, y_cols]
    #print(correlation)

    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation between Inputs (X) and Outputs (Y)")
    plt.show()

    related_inputs = correlation[correlation.abs() >= 0.1].dropna(how='all', axis=0).index.tolist()
    print('\n\n---------------------------')
    print(related_inputs)
    print('---------------------------\n\n')

    VAR_DICT_FILT = {}
    for key in related_inputs:
        VAR_DICT_FILT[key] = VAR_DICT[key]

    return VAR_DICT_FILT

if __name__=='__main__':
    
    dict_filt = analise_doe()

    pprint.pprint(dict_filt)
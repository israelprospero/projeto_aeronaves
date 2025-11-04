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

airplane_base = dt.standard_airplane('my_airplane_1')

def get_5perc_var(val):
    if val > 0:
        return val - 0.05 * val, val + 0.05 * val
    else:
        return val + 0.05 * val, val - 0.05 * val


airplane_ref = dt.standard_airplane('my_airplane_1')
VAR_DICT = {
    # 'var':            [lower bound, upper bound]
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

def update_airplane(airplane, x):
    for key, val in zip(VAR_NAMES, x):
        airplane[key] = val
    return airplane

def run_analysis(x):

    airplane = update_airplane(copy.deepcopy(airplane_base), x)
    dt.analyze(airplane, print_log=False, plot=False)
    
    airplane = aero.analise_aerodinamica(airplane, show_results=False)

    doe_out = [
        airplane['W0'],
        airplane['aero_CD_cruise']
    ]  

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
        
    MTOW_samples = np.zeros(n_samples)
    CD_cruise_samples = np.zeros(n_samples)
    for i in range(n_samples):

        doe_res = run_analysis(samples[i,:])
        
        MTOW = doe_res[0]
        CD_cruise = doe_res[1]

        MTOW_samples[i] = MTOW
        CD_cruise_samples[i] = CD_cruise
        
    data_dict = {var_name: samples[:, i] for i, var_name in enumerate(VAR_NAMES)}

    data_dict['MTOW'] = MTOW_samples
    data_dict['CD_cruise'] = CD_cruise_samples

    df = pd.DataFrame(data_dict)

    x_cols = VAR_NAMES
    y_cols = ['MTOW', 'CD_cruise']
    correlation = df[x_cols + y_cols].corr().loc[x_cols, y_cols]
    #print(correlation)

    #plt.figure(figsize=(6, 4))
    #sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)
    #plt.title("Correlation between Inputs (X) and Outputs (Y)")
    #plt.show()

    related_inputs = correlation[correlation.abs() >= 0.1].dropna(how='all', axis=0).index.tolist()
    print('\n\n---------------------------')
    print(related_inputs)
    print('---------------------------\n\n')

    VAR_DICT_FILT = {}
    for key in related_inputs:
        VAR_DICT_FILT[key] = VAR_DICT[key]

    return VAR_DICT_FILT
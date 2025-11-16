import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.optimize import minimize
import modules.designTool as dt
import copy
from pprint import pprint
import otimizacao.aerodinamica as aero
import otimizacao.pesos as pesos
import otimizacao.desempenho as desemp
import otimizacao.estabilidade as estab
import json
import matplotlib.pyplot as plt

ENABLE_REALTIME_PLOTS = False 

airplane_base = dt.standard_airplane('my_airplane_1')
xlist, flist, g_list = [], [], []
h_list = []
count = 0

def update_airplane(airplane, x):
    for key, val in zip(VAR_DICT_FILT, x):
        airplane[key] = val
    return airplane

def run_analysis(x):

    airplane = update_airplane(copy.deepcopy(airplane_base), x)
    dt.analyze(airplane, print_log=False, plot=False)
    
    # airplane = aero.analise_aerodinamica(airplane, show_results=False)

    f = airplane['W0']

    # default is g(x) >= 0
    g = [
        airplane['deltaS_wlan'],
        0.4 - airplane['SM_fwd'],
        airplane['SM_aft'] - 0.05,
        0.12 - airplane['SM_aft'],
        0.75 - airplane['CLv'],
        0.15 - airplane['frac_nlg_fwd'],
        airplane['frac_nlg_aft'] - 0.04,
        airplane['alpha_tipback'] * 180 / np.pi - 15,
        airplane['alpha_tailstrike'] * 180 / np.pi - 10,
        63 - airplane['phi_overturn'] * 180 / np.pi,
        (1 - airplane['x_tank_c_w']) - (airplane['c_tank_c_w'] + airplane['c_flap_c_wing']),
        (1 - airplane['x_tank_c_w']) - (airplane['c_tank_c_w'] + airplane['c_ail_c_wing']),
        0.85 - (airplane['b_flap_b_wing'] + airplane['b_ail_b_wing']),
        # airplane['SM_fwd'] - (airplane['SM_aft'] + 20),
        # (airplane['xcg_mlg'] - airplane['xm_w'])/airplane['cm_w'] - ((airplane['xnp'] - airplane['xm_w'])/airplane['cm_w'] + 0.08)
        0.005 - airplane['tank_excess'],
        
        
    ]
    
    h = [
        # airplane['tank_excess']
    ]
    
    global count
    count += 1
    print(f'Running Analysis ({count})')

    return f, g, h

def objfun(xn):
    x = denormalize(xn)
    f, g, h = run_analysis(x)

    xlist.append(x)
    flist.append(f)
    g_list.append(g)
    h_list.append(h)
    
    if ENABLE_REALTIME_PLOTS:
    
        line_f.set_xdata(range(len(flist)))
        line_f.set_ydata(flist)
        ax_f.relim()
        ax_f.autoscale_view()
        fig_f.canvas.draw()
        fig_f.canvas.flush_events()

        iterations = range(len(xlist))
        arr = np.array(xlist)  # (iterations, num_vars)

        for subplot_id, var_index in enumerate(tracked_indices):
            lines_x[subplot_id].set_xdata(iterations)
            lines_x[subplot_id].set_ydata(arr[:, var_index])
            ax_x[subplot_id].relim()
            ax_x[subplot_id].autoscale_view()

        fig_x.canvas.draw()
        fig_x.canvas.flush_events()
        
        # === Update constraint subplots ===
        iterations = range(len(g_list))
        arrg = np.array(g_list)  # (iters, num_g)

        for i in range(num_g):
            lines_g[i].set_xdata(iterations)
            lines_g[i].set_ydata(arrg[:, i])
            ax_g[i].relim()
            ax_g[i].autoscale_view()

        fig_g.canvas.draw()
        fig_g.canvas.flush_events()
    
    return f


def ineqconfun(xn):
    x = denormalize(xn)
    _, g, _ = run_analysis(x)
    return g

def eqconfun(xn):
    x = denormalize(xn)
    _, _, h = run_analysis(x)
    return h

def get_perc_var(val):
    perc = 0.1
    if val > 0:
        return val - perc * val, val + perc * val
    else:
        return val + perc * val, val - perc * val

airplane_ref = dt.standard_airplane('my_airplane_1')
VAR_DICT_FILT = {
    'S_w':              [70, 110],
    # 'AR_w':             [7, 11],
    'taper_w':          [0.2, 0.4],
    'sweep_w':          [20*np.pi/180, 30*np.pi/180],
    # 'dihedral_w':       [2*np.pi/180, 6.5*np.pi/180],
    'xr_w':             [10.6, 12],
    'zr_w':             list(get_perc_var(airplane_ref['zr_w'])),
    'tcr_w':            [0.12, 0.16],
    'tct_w':            [0.08, 0.11],
    # 'Cht':              [0.9, 1.15],
    'Lc_h':             [3, 4.7],
    # 'AR_h':             [3.5, 5.5],
    # 'taper_h':          [0.3, 0.5],
    # 'sweep_h':          [23*np.pi/180, 35*np.pi/180],
    # 'dihedral_h':       [3*np.pi/180, 10*np.pi/180],
    # 'zr_h':             [0.6, 0.9],
    # 'tcr_h':            [0.05, 0.15],
    # 'tct_h':            [0.05, 0.15],
    # 'Cvt':              [0.06, 0.11],
    'Lb_v':             [0.35, 0.51],
    # 'AR_v':             [1, 2],
    # 'taper_v':          [0.3, 0.6],
    # 'sweep_v':          [30*np.pi/180, 50*np.pi/180],
    # 'zr_v':             list(get_perc_var(airplane_ref['zr_v'])),
    'x_n':              [8.7, 14],
    'y_n':              [3, 5],
    'z_n':              list(get_perc_var(airplane_ref['z_n'])),
    # 'L_n':              [4, 4.5],
    # 'D_n':              [1.5, 2.3],
    'x_nlg':            [3, 5.5],
    'x_mlg':            [12.41, 15],
    'y_mlg':            [2, 4],
    # 'z_lg':             [-4, -2],
    'x_tailstrike':     list(get_perc_var(airplane_ref['x_tailstrike'])),
    'z_tailstrike':     list(get_perc_var(airplane_ref['z_tailstrike'])),
    'c_tank_c_w':       [0.45, 0.55],
    'c_flap_c_wing':    [0.2, 0.3],
    'b_flap_b_wing':    [0.55, 0.7],
    'c_ail_c_wing':     [0.2, 0.35],
    'b_ail_b_wing':     [0.3, 0.4]
}

VAR_NAMES_FILT = list(VAR_DICT_FILT.keys())

print('Starting optmization...')

x_min = np.array([VAR_DICT_FILT[k][0] for k in VAR_NAMES_FILT])
x_max = np.array([VAR_DICT_FILT[k][1] for k in VAR_NAMES_FILT])

x0_physical = np.array([airplane_base[k] for k in VAR_NAMES_FILT])

def normalize(x): return (x - x_min) / (x_max - x_min)

def denormalize(xn): return x_min + xn * (x_max - x_min)

x0_norm = normalize(x0_physical)
bounds_norm = [(0.0, 1.0) for _ in VAR_NAMES_FILT]

con_ineq = {'type': 'ineq', 'fun': ineqconfun}
con_eq   = {'type': 'eq', 'fun': eqconfun}
cons = [con_ineq, con_eq]

if ENABLE_REALTIME_PLOTS:
    VARS_TO_TRACK = [
        'S_w',
    ]

    plt.ion()

    fig_f, ax_f = plt.subplots()
    line_f, = ax_f.plot([], [], '-o')
    ax_f.set_xlabel('Iteration')
    ax_f.set_ylabel('Objective f')
    ax_f.set_title('Optimization Progress')
    plt.show()

    tracked_indices = [VAR_NAMES_FILT.index(v) for v in VARS_TO_TRACK]
    num_tracked = len(tracked_indices)

    fig_x, ax_x = plt.subplots(num_tracked, 1, figsize=(6, 2*num_tracked), sharex=True)
    if num_tracked == 1:
        ax_x = [ax_x]

    lines_x = []

    for i, varname in enumerate(VARS_TO_TRACK):
        line_var, = ax_x[i].plot([], [], '-')
        lines_x.append(line_var)
        ax_x[i].set_ylabel(varname)
        
        ymin, ymax = VAR_DICT_FILT[varname]
        ax_x[i].set_ylim([ymin, ymax])
        
    ax_x[-1].set_xlabel('Iteration')
    fig_x.suptitle('Evolution of Selected Variables')
    plt.tight_layout()
    plt.show()
    
    # === Constraint plots (each constraint in its own subplot) ===
    num_g = len(run_analysis(denormalize(x0_norm))[1])

    fig_g, ax_g = plt.subplots(num_g, 1, figsize=(6, 2*num_g), sharex=True)
    if num_g == 1:
        ax_g = [ax_g]

    lines_g = []

    for i in range(num_g):
        line, = ax_g[i].plot([], [], '-')
        lines_g.append(line)
        ax_g[i].set_ylabel(f"g[{i}]")
        ax_g[i].axhline(0.0, linestyle='--')  # constraint threshold

    ax_g[-1].set_xlabel('Iteration')
    fig_g.suptitle('Constraint Evolution')
    plt.tight_layout()
    plt.show()

result = minimize(objfun, x0_norm, constraints=cons, bounds=bounds_norm, method='SLSQP')
print(result)

xopt_norm = result.x
xopt = denormalize(xopt_norm)

airplane_opt = update_airplane(copy.deepcopy(airplane_base), xopt)
airplane_opt = dt.analyze(airplane_opt, print_log=False, plot=False)
# dt.plot3d(airplane_opt)

airplane_base = dt.analyze(airplane_base, print_log=False, plot=False)

# Print each key comparison
# all_keys = sorted(set(airplane_base) | set(airplane_opt))
# for key in all_keys:
#     val1 = airplane_base.get(key, '—')
#     val2 = airplane_opt.get(key, '—')
#     print(f"{key}: {val1}\t {val2}")
    
with open("airplane_opt.json", "w") as file:
    json.dump(airplane_opt, file, indent=4)

# input()
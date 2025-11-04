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
from otimizacao.doe import analise_doe

airplane_base = dt.standard_airplane('my_airplane_1')
xlist, flist, g_hist = [], [], []
h_list = []

def update_airplane(airplane, x):
    for key, val in zip(VAR_DICT_FILT, x):
        airplane[key] = val
    return airplane

def run_analysis(x):

    airplane = update_airplane(copy.deepcopy(airplane_base), x)
    dt.analyze(airplane, print_log=False, plot=False)
    
    airplane = aero.analise_aerodinamica(airplane, show_results=False)

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

VAR_DICT_FILT = analise_doe()
pprint(VAR_DICT_FILT)
VAR_NAMES_FILT = list(VAR_DICT_FILT.keys())

input('Press ENTER to start optimization')
print('Starating optmization...')

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

result = minimize(objfun, x0_norm, constraints=cons, bounds=bounds_norm, method='SLSQP')
xopt_norm = result.x
xopt = denormalize(xopt_norm)

# print(result)

airplane_opt = update_airplane(copy.deepcopy(airplane_base), xopt)

# Print each key comparison
all_keys = sorted(set(airplane_base) | set(airplane_opt))
for key in all_keys:
    val1 = airplane_base.get(key, '—')
    val2 = airplane_opt.get(key, '—')
    print(f"{key}: {val1}\t {val2}")

airplane_opt = dt.analyze(airplane_opt, print_log=False, plot=False)
dt.plot3d(airplane_opt)
# pprint(airplane_opt)

airplane_opt = aero.analise_aerodinamica(airplane_opt, show_results=False)
#pesos.analise_pesos(airplane_opt)
#desemp.analise_desempenho(airplane_opt)
#estab.analise_estabilidade(airplane_opt)
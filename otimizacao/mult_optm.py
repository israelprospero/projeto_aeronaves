import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import modules.designTool as dt
import copy
from pprint import pprint
import otimizacao.aerodinamica as aero
import otimizacao.pesos as pesos
import otimizacao.desempenho as desemp
import otimizacao.estabilidade as estab


airplane_base = dt.standard_airplane('my_airplane_1')

def get_perc_var(val):
    perc = 0.1
    if val > 0:
        return val - perc * val, val + perc * val
    else:
        return val + perc * val, val - perc * val

airplane_ref = dt.standard_airplane('my_airplane_1')
VAR_DICT_FILT = {
    'S_w':              [70, 110],
    'AR_w':             [7, 11],
    'taper_w':          [0.2, 0.4],
    'sweep_w':          [20*np.pi/180, 30*np.pi/180],
    # 'dihedral_w':       [2*np.pi/180, 6.5*np.pi/180],
    'xr_w':             list(get_perc_var(airplane_ref['xr_w'])),
    # 'zr_w':             list(get_perc_var(airplane_ref['zr_w'])),
    'tcr_w':            [0.12, 0.16],
    'tct_w':            [0.08, 0.11],
    'Cht':              [0.9, 1.15],
    # 'Lc_h':             [3, 4.3],
    # 'AR_h':             [3.5, 5.5],
    # 'taper_h':          [0.3, 0.5],
    # 'sweep_h':          [23*np.pi/180, 35*np.pi/180],
    # 'dihedral_h':       [3*np.pi/180, 10*np.pi/180],
    # 'zr_h':             [0.6, 0.9],
    # 'tcr_h':            [0.05, 0.15],
    # 'tct_h':            [0.05, 0.15],
    'Cvt':              [0.06, 0.11],
    # 'Lb_v':             [0.35, 0.45],
    # 'AR_v':             [1, 2],
    # 'taper_v':          [0.3, 0.6],
    # 'sweep_v':          [30*np.pi/180, 50*np.pi/180],
    # 'zr_v':             list(get_perc_var(airplane_ref['zr_v'])),
    'x_n':              [airplane_ref['xr_w'] - 2.5, airplane_ref['xr_w'] + 4],
    'y_n':              list(get_perc_var(airplane_ref['y_n'])),
    'z_n':              list(get_perc_var(airplane_ref['z_n'])),
    'L_n':              [4, 4.5],
    'D_n':              [1.5, 2.3],
    'x_nlg':            [2.5, 5.5],
    'x_mlg':            [airplane_ref['xr_w'] + 1.7, airplane_ref['xr_w'] + 3.9],
    'y_mlg':            [2, 4],
    'z_lg':             [-4, -2],
    'x_tailstrike':     list(get_perc_var(airplane_ref['x_tailstrike'])),
    'z_tailstrike':     list(get_perc_var(airplane_ref['z_tailstrike'])),
    'c_tank_c_w':       [0.45, 0.55],
    'c_flap_c_wing':    [0.2, 0.3],
    'b_flap_b_wing':    [0.55, 0.7],
    'c_ail_c_wing':     [0.2, 0.35],
    'b_ail_b_wing':     [0.3, 0.4]
}

pprint(VAR_DICT_FILT)
VAR_NAMES_FILT = list(VAR_DICT_FILT.keys())

x_min = np.array([VAR_DICT_FILT[k][0] for k in VAR_NAMES_FILT])
x_max = np.array([VAR_DICT_FILT[k][1] for k in VAR_NAMES_FILT])
x0_physical = np.array([airplane_base[k] for k in VAR_NAMES_FILT])

def normalize(x):
    return (x - x_min) / (x_max - x_min)

def denormalize(xn):
    return x_min + xn * (x_max - x_min)

def update_airplane(airplane, x):
    for key, val in zip(VAR_DICT_FILT, x):
        airplane[key] = val
    return airplane


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as minimize_pymoo
from pymoo.core.problem import ElementwiseProblem
import matplotlib.pyplot as plt

class AirplaneMOO(ElementwiseProblem):

    def __init__(self):
        n_var = len(VAR_NAMES_FILT)
        n_ineq = 12
        n_eq   = 2
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=n_ineq + n_eq,
            xl=np.zeros(n_var),
            xu=np.ones(n_var)
        )

    def _evaluate(self, xn, out, *args, **kwargs):

        x = denormalize(xn)

        airplane = update_airplane(copy.deepcopy(airplane_base), x)

        airplane = dt.analyze(airplane, print_log=False, plot=False)
        airplane = aero.analise_aerodinamica(airplane, show_results=False)

        F1 = airplane['W0']
        F2 = -airplane['aero_CD_cruise']

        # default is <= 0
        g = [
            - (airplane['deltaS_wlan']),
            - (0.4 - airplane['SM_fwd']),
            - (airplane['SM_aft'] - 0.05),
            - (0.75 - airplane['CLv']),
            - (0.15 - airplane['frac_nlg_fwd']),
            - (airplane['frac_nlg_aft'] - 0.04),
            - (airplane['alpha_tipback'] * 180 / np.pi - 15),
            - (airplane['alpha_tailstrike'] * 180 / np.pi - 10),
            - (63 - airplane['phi_overturn'] * 180 / np.pi),
            - (0.9 - (airplane['c_tank_c_w'] + airplane['c_flap_c_wing'])),
            - (0.9 - (airplane['c_tank_c_w'] + airplane['c_ail_c_wing'])),
            - (0.8 - (airplane['b_flap_b_wing'] + airplane['b_ail_b_wing']))
        ]

        # Equality constraints converted to |h| <= tolerance
        h = [
            airplane['tank_excess'],
        ]
        tol = 1e-4
        h_as_ineq = [abs(val) - tol for val in h]

        out["F"] = [F1, F2]
        out["G"] = g + h_as_ineq


problem = AirplaneMOO()
algorithm = NSGA2(pop_size=80, eliminate_duplicates=True)

res = minimize_pymoo(
    problem,
    algorithm,
    ('n_gen', 60),
    verbose=True,
)

plt.figure()
plt.scatter(res.F[:, 0], res.F[:, 1])
plt.xlabel('W0')
plt.ylabel('-CLmax')
plt.title('Pareto Front: Objective Space')
plt.tight_layout()

plt.figure()
plt.scatter(res.X[:, 0], res.X[:, 1])
plt.xlabel('x1 (normalized)')
plt.ylabel('x2 (normalized)')
plt.title('Pareto Front: Design Variable Space')
plt.tight_layout()

plt.show()

pareto_physical = np.array([denormalize(x) for x in res.X])

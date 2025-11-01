import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.optimize import minimize
import auxmod as am
import matplotlib.pyplot as plt
import modules.designTool as dt
import copy

airplane_base = dt.standard_airplane('my_airplane_1')

xlist, flist = [], []
g_hist, h_hist = [], []

def run_analysis(x):
    
    airplane = copy.deepcopy(airplane_base)

    airplane['S_w'] = x[0] # Wing area [m2]
    airplane['AR_w'] = x[1]  # Wing aspect ratio
    
    dt.analyze(airplane, print_log=False, plot=False)

    tank_excess = airplane['tank_excess']
    CLv = airplane['CLv']
    SM_fwd = airplane['SM_fwd']

    # Objetivo: minimizar o tank_excess
    f = tank_excess

    # Restricoes de desigualdade (g(x) >= 0)
    g1 = 0.75 - CLv

    # Restricao de igualdade (h(x) = 0)
    h1 = SM_fwd - 0.4            

    return f, [g1], [h1]

def objfun(x):
    
    f, g, h = run_analysis(x)
    xlist.append(x)
    flist.append(f)
    g_hist.append(g)
    h_hist.append(h)
    return f

def ineqconfun(x):
    _, g, _ = run_analysis(x)
    return g

def eqconfun(x):
    _, _, h = run_analysis(x)
    return h

con_ineq = {'type': 'ineq', 'fun': ineqconfun}
con_eq   = {'type': 'eq', 'fun': eqconfun}
cons = [con_ineq, con_eq]

# Limites das variaveis de projeto
bounds = [[80, 100],   # S_w
          [8, 12]]     # AR

x0 = np.array([90, 9]) 

result = minimize(objfun, x0, constraints=cons, bounds=bounds, method='SLSQP')
xopt = result.x
print(result)

airplane_opt = copy.deepcopy(airplane_base)
airplane_opt['S_w'] = xopt[0]
airplane_opt['AR_w'] = xopt[1]
dt.analyze(airplane_opt, print_log=False, plot=False)

fig = plt.figure(figsize=(8, 8))

plt.subplot(311)
plt.plot([xi[0] for xi in xlist], 'o-', label='x[0]')
plt.plot([xi[1] for xi in xlist], 'o-', label='x[1]')
plt.ylabel('x', fontsize=14)
plt.legend()

plt.subplot(312)
plt.plot(flist, 'o-')
plt.ylabel('Tank Excess', fontsize=14)

plt.subplot(313)
for i in range(len(g_hist[0])):  # cada restricao separada
    plt.plot([g[i] for g in g_hist], 'o-', label=f'g{i+1}')
for i in range(len(h_hist[0])):  # restricoes de igualdade
    plt.plot([h[i] for h in h_hist], 's--', label=f'h{i+1}')
plt.plot([0, len(g_hist) - 1], [0, 0], 'gray', linewidth=0.5)
plt.ylabel('g (-), h (--)', fontsize=14)
plt.xlabel('Num. Iter.', fontsize=14)
plt.legend()
plt.tight_layout()

plt.show()

# xk = xlist.copy()
# fig = plt.figure()
# ax = plt.gca()
# am.plot_contour(objfun, ax,
#                 xmin=-3*1.3, xmax=3*1.3, ymin=-3, ymax=3, zmin=10**-3, zmax=10**(-0.1))

# confun = lambda x: ineqconfun(x)[0]
# am.plot_contour(confun, ax,
#                 xmin=-3*1.3, xmax=3*1.3, ymin=-3, ymax=3, zmin=0, zmax=0, nlevels=None)
# confun = lambda x: ineqconfun(x)[1]
# am.plot_contour(confun, ax,
#                 xmin=-3*1.3, xmax=3*1.3, ymin=-3, ymax=3, zmin=0, zmax=0, nlevels=None)
# confun = lambda x: eqconfun(x)
# am.plot_contour(confun, ax,
#                 xmin=-3*1.3, xmax=3*1.3, ymin=-3, ymax=3, zmin=0, zmax=0, nlevels=None)

# am.plot_path(ax, xk, xopt=xopt)

# plt.show()
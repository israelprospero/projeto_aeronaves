from modules import designTool as dt
from modules import utils as m
import numpy as np
import matplotlib.pyplot as plt
from modules.utils import print_fuel_table

gravity = dt.gravity
ft2m = dt.ft2m
nm2m = dt.nm2m
pi = np.pi

airplane = dt.standard_airplane('my_airplane_1')
dt.geometry(airplane)

# dt.plot3d(airplane)

# 1 - MTOW
W0, W_empty, W_fuel, _ = dt.weight(airplane['W0_guess'], airplane['T0_guess'], airplane)

print('Weights in kgf:')
print('W0: %d '%(W0/dt.gravity))
print('W_empty : %d '%(W_empty/dt.gravity))
print('W_fuel : %d '%(W_fuel/dt.gravity))
print('W_payload : %d '%(airplane['W_payload']/dt.gravity))
print('W_crew : %d '%(airplane['W_crew']/dt.gravity))

for key in ['W_w','W_h','W_v','W_f','W_nlg','W_mlg','W_eng','W_allelse']:
    print(key +': %d '%(airplane[key]/dt.gravity))

# 2 - MTOW x AR_w
ar_w = np.arange(6.0, 14.0,0.01) # varia o alongamento de 6 a 14 no passo de 0.01 e armazena no dicionário
m.plot_W0_x_ar_w(ar_w, airplane, 1)

# 5 - Pie Chart

# 6 - Table
print_fuel_table(airplane)

# 7 - L/D and TSFC

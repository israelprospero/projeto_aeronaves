from modules import designTool as dt
from modules import utils as m
import numpy as np
import matplotlib.pyplot as plt
from modules.utils import print_fuel_table

gravity = dt.gravity
ft2m = dt.ft2m
nm2m = dt.nm2m
pi = np.pi

airplane_1 = dt.standard_airplane('my_airplane_1')
dt.geometry(airplane_1)

# ------------------------------------------------------------------------------------------------
### WEEK 5 ###
### Questions 2 and 3 ###

# ar_w = airplane_1['AR_w'] # cria a variável 'aspect ratio wing' (alongamento da asa) 
ar_w = np.arange(6.0, 14.0,0.01) # varia o alongamento de 6 a 14 no passo de 0.01 e armazena no dicionário

## Plot
# W0 vs ar_w
#m.plot_W0_x_ar_w(ar_w, airplane_1, 1)

### WEEK 5 ###
## Weight estimation
W0_guess = 40000 * dt.gravity
T0_guess = 0.3 * W0_guess

W0, W_empty, W_fuel, Mf_cruise = dt.weight(W0_guess, T0_guess, airplane_1)

print_fuel_table(airplane_1)
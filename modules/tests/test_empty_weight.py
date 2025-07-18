import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import designTool as dt
import numpy as np

airplane = dt.standard_airplane('fokker100')
dt.geometry(airplane)

Mach = 0.73000000000000
altitude = 10668.00000000000000
CL = 0.50000000000000
n_engines_failed = 0.00000000000000
highlift_config = 'clean'
lg_down = 0.00000000000000
h_ground = 0.00000000000000

W0_guess = 467500.00000000000000
T0_guess = 140250.00000000000000

CD, CLmax, dragDict = dt.aerodynamics( airplane , Mach , altitude , CL ,
n_engines_failed = n_engines_failed ,highlift_config = highlift_config ,
lg_down = lg_down , h_ground = h_ground, W0_guess=W0_guess )

W_empty = dt.empty_weight ( W0_guess , T0_guess , airplane )
print('W_empty = ', W_empty)
print('W_allelse =', airplane['W_allelse'])
print('W_eng =', airplane['W_eng'])
print('W_f =', airplane['W_f'])
print('W_h =', airplane['W_h'])
print('W_mlg =', airplane['W_mlg'])
print('W_nlg =', airplane['W_nlg'])
print('W_v =', airplane['W_v'])
print('W_w =', airplane['W_w'])
print('x_CG_empty =', airplane['xcg_empty'])
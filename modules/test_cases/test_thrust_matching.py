import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import designTool as dt
import numpy as np

airplane = dt.standard_airplane('fokker100')
dt.geometry(airplane)

W0_guess = 467500.00000000000000
T0_guess = 140250.00000000000000

dt.thrust_matching(W0_guess, T0_guess, airplane)

print("airplane['CLmaxTO'] = ",airplane['CLmaxTO'])
print("airplane['T0'] = ",airplane['T0'])
print("airplane['T0vec'] = ",airplane['T0vec'])
print("airplane['W0'] = ",airplane['W0'])
print("airplane['W_empty'] = ",airplane['W_empty'])
print("airplane['W_fuel'] = ",airplane['W_fuel'])
print("airplane['deltaS_wlan'] = ",airplane['deltaS_wlan'])

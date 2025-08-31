import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import designTool as dt
import numpy as np

airplane = dt.standard_airplane('fokker100')

dt.geometry(airplane)

W0_guess = 467500.00000000000000
T0_guess = 140250.00000000000000

W0 , W_empty , W_fuel , W_cruise = dt. weight (W0_guess , T0_guess , airplane )

print (" W0 = ",W0)
print (" W_empty = ", W_empty )
print (" W_fuel = ", W_fuel )
print (" W_cruise = ", W_cruise )
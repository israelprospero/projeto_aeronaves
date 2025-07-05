import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import designTool as dt
import numpy as np

airplane = dt.standard_airplane('fokker100')
dt.geometry(airplane)

Mach = 0.73000000000000
altitude = 10668.00000000000000

C, kT = dt.engineTSFC (Mach , altitude , airplane )

print ("C = ", C)
print ("kT = ", kT)
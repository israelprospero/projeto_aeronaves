'''This file aims to analyse the aerodynamics of the aircraft in order to confirm that
the changes made for stability purposes did not compromise the aerodynamics.

Essentially, this is just a copy of the test_aerodynamics.py file using the team's aircraft'''

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import designTool as dt
import numpy as np
import pprint
import utils as m

airplane = dt.standard_airplane('my_airplane_1')
dt.geometry(airplane)

W0_guess = 50150*dt.gravity
T0_guess = 0.3*W0_guess 

dt.thrust_matching(W0_guess, T0_guess, airplane)
dt.balance(airplane)

# Cruise
Mach = 0.8
altitude = 10668
n_engines_failed = 0
highlift_config = 'clean'
lg_down = 0
h_ground = 0

rho = dt.atmosphere(altitude)[2]
V = Mach * m.get_a(altitude)
CL = 0.95 * airplane['W0'] / (0.5 * rho * V**2 * airplane['S_w'])

CD, CLmax, dragDict = dt.aerodynamics(airplane, Mach, altitude, CL, n_engines_failed = n_engines_failed,
highlift_config = highlift_config, lg_down = lg_down, h_ground = h_ground)

print (" Cruise conditions")
print (" CD = ",CD)
print (" CLmax = ", CLmax )
print (" dragDict = " + pprint . pformat ( dragDict ))
print ("")

# Takeoff/Climb
Mach = 0.3
altitude = 0
CL = 1.527
n_engines_failed = 1
highlift_config = 'takeoff'
lg_down = 0
h_ground = 0

CD, CLmax, dragDict = dt.aerodynamics(airplane, Mach, altitude, CL, n_engines_failed = n_engines_failed, highlift_config = highlift_config, lg_down = lg_down, h_ground = h_ground)

print (" Takeoff climb conditions ")
print (" CD = ",CD)
print (" CLmax = ", CLmax )
print (" dragDict = " + pprint . pformat ( dragDict ))
print ("")

# Landing
Mach = 0.3
altitude = 0
CL = 1.53
n_engines_failed = 0
highlift_config = 'landing'
lg_down = 1
h_ground = 10.7

CD, CLmax, dragDict = dt.aerodynamics(airplane, Mach, altitude, CL, n_engines_failed = n_engines_failed, highlift_config = highlift_config, lg_down = lg_down, h_ground = h_ground)

print (" Landing approach conditions ")
print (" CD = ",CD)
print (" CLmax = ", CLmax )
print (" dragDict = " + pprint . pformat ( dragDict ))
print ("")
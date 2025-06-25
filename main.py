import designTool as dt
import numpy as np
import pprint

gravity = dt.gravity
ft2m = dt.ft2m
nm2m = dt.nm2m
pi = np.pi


# AIRPLANE 1
print('==============================')
print('AIRPLANE 1')
print('==============================')
airplane_1 = dt.standard_airplane('my_airplane_1')
dt.geometry(airplane_1)
#dt.plot3d(airplane_1)

# Cruise
dt.geometry(airplane_1)
_, _, dragDict_cruise = dt.aerodynamics(airplane_1, Mach=0.73, altitude=10668, CL=0.5, W0_guess=airplane_1['W0_guess'], n_engines_failed=0, highlift_config='clean', lg_down=0, h_ground=0)

print('CRUISE: \n' + pprint.pformat(dragDict_cruise))


# Takeoff
dt.geometry(airplane_1)
_, _, dragDict_TO = dt.aerodynamics(airplane_1, Mach=0.3, altitude=0, CL=0.5, W0_guess=airplane_1['W0_guess'], n_engines_failed=1, highlift_config='takeoff', lg_down=0, h_ground=0)

print('TAKEOFF: \n' + pprint.pformat(dragDict_TO))

# Landing
dt.geometry(airplane_1)
_, _, dragDict_landing = dt.aerodynamics(airplane_1, Mach=0.3, altitude=0, CL=0.5, W0_guess=airplane_1['W0_guess'], n_engines_failed=0, highlift_config='landing', lg_down=1, h_ground=10.668)

print('LANDING: \n' + pprint.pformat(dragDict_landing))

print('==============================')



# AIRPLANE 2
print('==============================')
print('AIRPLANE 2')
print('==============================')
airplane_2 = dt.standard_airplane('my_airplane_2')
dt.geometry(airplane_2)
#dt.plot3d(airplane_2)

# Cruise
dt.geometry(airplane_2)
_, _, dragDict_cruise = dt.aerodynamics(airplane_2, Mach=0.73, altitude=10668, CL=0.5, W0_guess=airplane_2['W0_guess'], n_engines_failed=0, highlift_config='clean', lg_down=0, h_ground=0)

print('CRUISE: \n' + pprint.pformat(dragDict_cruise))


# Takeoff
dt.geometry(airplane_2)
_, _, dragDict_TO = dt.aerodynamics(airplane_2, Mach=0.3, altitude=0, CL=0.5, W0_guess=airplane_2['W0_guess'], n_engines_failed=1, highlift_config='takeoff', lg_down=0, h_ground=0)

print('TAKEOFF: \n' + pprint.pformat(dragDict_TO))

# Landing
dt.geometry(airplane_2)
_, _, dragDict_landing = dt.aerodynamics(airplane_2, Mach=0.3, altitude=0, CL=0.5, W0_guess=airplane_2['W0_guess'], n_engines_failed=0, highlift_config='landing', lg_down=1, h_ground=10.668)

print('LANDING: \n' + pprint.pformat(dragDict_landing))

print('==============================')
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import designTool as dt
import numpy as np
import numpy as np
import pprint

airplane = dt. standard_airplane ('fokker100')
# Execute the geometry function
dt. geometry ( airplane )
# Cruise conditions for aerodynamic analysis
Mach = 0.73000000000000
altitude = 10668.00000000000000
CL = 0.50000000000000
n_engines_failed = 0.00000000000000
highlift_config = 'clean'
lg_down = 0.00000000000000
h_ground = 0.00000000000000
# Execute the aerodynamic analysis
CD , CLmax , dragDict = dt. aerodynamics ( airplane , Mach , altitude , CL , n_engines_failed = n_engines_failed ,
highlift_config = highlift_config , lg_down = lg_down , h_ground = h_ground )
# Print results
print (" Cruise conditions")
print (" CD = ",CD)
print (" CLmax = ", CLmax )
print (" dragDict = " + pprint . pformat ( dragDict ))
print ("")
# Takeoff climb conditions for aerodynamic analysis
Mach = 0.30000000000000
altitude = 0.00000000000000
CL = 1.52777777777778
n_engines_failed = 1.00000000000000
highlift_config = 'takeoff'
lg_down = 0.00000000000000
h_ground = 0.00000000000000
# Execute the aerodynamic analysis
CD , CLmax , dragDict = dt. aerodynamics ( airplane , Mach , altitude , CL ,
n_engines_failed = n_engines_failed ,
highlift_config = highlift_config ,
lg_down = lg_down , h_ground = h_ground )
# Print results
print (" Takeoff climb conditions ")
print (" CD = ",CD)
print (" CLmax = ", CLmax )
print (" dragDict = " + pprint . pformat ( dragDict ))
print ("")
# Landing approach conditions for aerodynamic analysis
Mach = 0.30000000000000
altitude = 0.00000000000000
CL = 1.53846153846154

n_engines_failed = 0.00000000000000
highlift_config = 'landing'
lg_down = 1.00000000000000
h_ground = 10.66800000000000
# Execute the aerodynamic analysis
CD , CLmax , dragDict = dt. aerodynamics ( airplane , Mach , altitude , CL ,
n_engines_failed = n_engines_failed ,
highlift_config = highlift_config ,
lg_down = lg_down , h_ground = h_ground )
# Print results
print (" Landing approach conditions ")
print (" CD = ",CD)
print (" CLmax = ", CLmax )
print (" dragDict = " + pprint . pformat ( dragDict ))
print ("")
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import designTool as dt
import numpy as np
import pprint

airplane = dt . standard_airplane ('fokker100')
dt . geometry ( airplane )

W0_guess = 467500.00000000000000
T0_guess = 140250.00000000000000

dt . thrust_matching ( W0_guess , T0_guess , airplane )

dt . doc ( airplane , CEF =6.0 , plot = True )

print (" airplane ['DOC'] = " , airplane ['DOC'])
print (" airplane ['DOC_breakdown'] = " , airplane ['DOC_breakdown'])
import designTool as dt
import numpy
import pprint

# You have two options to define inputs.
# 1. Load a sample case already defined in designTools.py:
# airplane = dt.standard_airplane('fokker100')

# 2. Load another aircraft with student-defined data
# (uncomment the line below and change the dictionary parameters within designTools.py)
airplane = dt.standard_airplane('my_airplane_2')

# Execute the geometry function
dt.geometry(airplane)

# Print updated dictionary
print('airplane = ' + pprint.pformat(airplane))

# Generate 3D plot
dt.plot3d(airplane)

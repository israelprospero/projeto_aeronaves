from modules import designTool as dt
from modules import utils as m

airplane = dt.standard_airplane('my_airplane_1')
dt.geometry(airplane)

W0_guess = 50150*dt.gravity
T0_guess = 0.3*W0_guess 

dt.thrust_matching(W0_guess, T0_guess, airplane)
dt.balance(airplane)

# Print results
print("airplane['xcg_fwd'] = ",airplane['xcg_fwd'])
print("airplane['xcg_aft'] = ",airplane['xcg_aft'])
print("airplane['xnp'] = ",airplane['xnp'])
print("airplane['SM_fwd'] = ",airplane['SM_fwd'])
print("airplane['SM_aft'] = ",airplane['SM_aft'])
print("airplane['tank_excess'] = ",airplane['tank_excess'])
print("airplane['V_maxfuel'] = ",airplane['V_maxfuel'])
print("airplane['CLv'] = ",airplane['CLv'])
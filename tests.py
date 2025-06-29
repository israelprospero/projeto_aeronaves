import modules.designTool as dt
import numpy as np
import pprint

gravity = dt.gravity
ft2m = dt.ft2m
nm2m = dt.nm2m
pi = np.pi

# geometry() TEST
airplane = dt.standard_airplane('fokker100')
dt.geometry(airplane)
dt.plot3d(airplane)

# aerodynamics() TEST
inputs = {
'S_w' : 93.5, # Wing area [m2] - From Obert's paper
'AR_w' : 8.32, # Wing aspect ratio
'taper_w' : 0.25, # Wing taper ratio
'sweep_w' : 15.76*pi/180, # Wing sweep [rad]
'dihedral_w' : 3*pi/180, # Wing dihedral [rad]
'xr_w' : 13.38, # Longitudinal position of the wing (with respect to the fuselage nose) [m]
'zr_w' : -0.99, # Vertical position of the wing (with respect to the fuselage nose) [m]
'tcr_w' : 0.123, # t/c of the root section of the wing
'tct_w' : 0.096, # t/c of the tip section of the wing
'Cht' : 0.976, # Horizontal tail volume coefficient
'Lc_h' : 4.23, # Non-dimensional lever of the horizontal tail (lever/wing_mac)
'AR_h' : 4.52, # HT aspect ratio
'taper_h' : 0.44, # HT taper ratio
'sweep_h' : 26*np.pi/180, # HT sweep [rad]
'dihedral_h' : 2*np.pi/180, # HT dihedral [rad]
'zr_h' : 4.76, # Vertical position of the HT [m]
'tcr_h' : 0.1, # t/c of the root section of the HT
'tct_h' : 0.1, # t/c of the tip section of the HT
'eta_h' : 1.0, # Dynamic pressure factor of the HT
'Cvt' : 0.068, # Vertical tail volume coefficient
'Lb_v' : 0.469, # Non-dimensional lever of the vertical tail (lever/wing_span)
'AR_v' : 1.03, # VT aspect ratio
'taper_v' : 0.73, # VT taper ratio
'sweep_v' : 39*np.pi/180, # VT sweep [rad]
'zr_v' : 1.39, # Vertical position of the VT [m]
'tcr_v' : 0.1, # t/c of the root section of the VT
'tct_v' : 0.1, # t/c of the tip section of the VT
'L_f' : 31.6, # Fuselage length [m]
'D_f' : 3.28, # Fuselage diameter [m]
'x_n' : 21.4, # Longitudinal position of the nacelle frontal face [m]
'y_n' : 3.01, # Lateral position of the nacelle centerline [m]
'z_n' : 0.45, # Vertical position of the nacelle centerline [m]
'L_n' : 4.91, # Nacelle length [m]
'D_n' : 1.69, # Nacelle diameter [m]
'n_engines' : 2, # Number of engines
'n_engines_under_wing' : 0, # Number of engines installed under the wing
'engine' : {'model' : 'Howe turbofan', # Check engineTSFC function for options
    #'model' : 'Raymer turbofan', # Check engineTSFC function for options
    'BPR' : 3.04, # Engine bypass ratio
    'Cbase' : 0.58/3600, # I adjusted this value by hand to match TSFC=0.70 at cruise (This is the value I fou
    },
'x_nlg' : 3.7, # Longitudinal position of the nose landing gear [m]
'x_mlg' : 17.4, # Longitudinal position of the main landing gear [m]
'y_mlg' : 2.47, # Lateral position of the main landing gear [m]
'z_lg' : -2.53, # Vertical position of the landing gear [m]
'x_tailstrike' : 23.4, # Longitudinal position of critical tailstrike point [m]
'z_tailstrike' : -1.54, # Vertical position of critical tailstrike point [m]
'c_tank_c_w' : 0.4, # Fraction of the wing chord occupied by the fuel tank
'x_tank_c_w' : 0.2, # Fraction of the wing chord where fuel tank starts
'b_tank_b_w_start' : 0.0, # Fraction of the wing semi-span where fuel tank starts
'b_tank_b_w_end' : 0.95, # Fraction of the wing semi-span where fuel tank ends
'clmax_w' : 1.8, # Maximum lift coefficient of wing airfoil
'k_korn' : 0.91, # Airfoil technology factor for Korn equation (wave drag)
'flap_type' : 'double slotted', # Flap type
'c_flap_c_wing' : 0.30, # Fraction of the wing chord occupied by flaps
'b_flap_b_wing' : 0.60, # Fraction of the wing span occupied by flaps (including fuselage portion)
'slat_type' : None, # Slat type
'c_slat_c_wing' : 0.0, # Fraction of the wing chord occupied by slats
'b_slat_b_wing' : 0.0, # Fraction of the wing span occupied by slats
'c_ail_c_wing' : 0.27, # Fraction of the wing chord occupied by aileron
'b_ail_b_wing' : 0.34, # Fraction of the wing span occupied by aileron
'h_ground' : 35.0*ft2m, # Distance to the ground for ground effect computation [m]
'k_exc_drag' : 0.03, # Excrescence drag factor
'winglet' : False, # Add winglet
'altitude_takeoff' : 0.0, # Altitude for takeoff computation [m] - From Obert's paper
'distance_takeoff' : 2050.0, # Required takeoff distance [m] - From Obert's paper
'deltaISA_takeoff' : 15.0, # Variation from ISA standard temperature [oC] - From Obert's paper
'altitude_landing' : 0.0, # Altitude for landing computation [m]
'distance_landing' : 1340.0, # Required landing distance [m]
'deltaISA_landing' : 0.0, # Variation from ISA standard temperature [oC]
'MLW_frac' : 40100/43090, # Max Landing Weight / Max Takeoff Weight - From Obert's paper
'altitude_cruise' : 35000*ft2m, # Cruise altitude [m] - From Obert's paper
'Mach_cruise' : 0.73, # Cruise Mach number - From Obert's paper
'range_cruise' : 1310*nm2m, # Cruise range [m] - From Obert's paper
'loiter_time' : 45*60, # Loiter time [s]
'altitude_altcruise' : 4572, # Alternative cruise altitude [m]
'Mach_altcruise' : 0.4, # Alternative cruise Mach number
'range_altcruise' : 200*nm2m, # Alternative cruise range [m]
'W_payload' : 107*91*gravity, # Payload weight [N]
'xcg_payload' : 14.4, # Longitudinal position of the Payload center of gravity [m]
'W_crew' : 5*91*gravity, # Crew weight [N]
'xcg_crew' : 2.5, # Longitudinal position of the Crew center of gravity [m]
'block_range' : 400*nm2m, # Block range [m]
'block_time' : (1.0 + 2*40/60)*3600, # Block time [s]
'n_captains' : 1, # Number of captains in flight
'n_copilots' : 1, # Number of copilots in flight
'rho_fuel' : 804, # Fuel density kg/m3 (This is Jet A-1)
'W0_guess' : 40000*gravity # Guess for MTOW
}

# Cruise
dt.geometry(inputs)
_, _, dragDict_cruise = dt.aerodynamics(inputs, Mach=0.73, altitude=10668, CL=0.5, W0_guess=inputs['W0_guess'], n_engines_failed=0, highlift_config='clean', lg_down=0, h_ground=0)

print('CRUISE: \n' + pprint.pformat(dragDict_cruise))


# Takeoff
dt.geometry(inputs)
_, _, dragDict_TO = dt.aerodynamics(inputs, Mach=0.3, altitude=0, CL=0.5, W0_guess=inputs['W0_guess'], n_engines_failed=1, highlift_config='takeoff', lg_down=0, h_ground=0)

print('TAKEOFF: \n' + pprint.pformat(dragDict_TO))

# Landing
dt.geometry(inputs)
_, _, dragDict_landing = dt.aerodynamics(inputs, Mach=0.3, altitude=0, CL=0.5, W0_guess=inputs['W0_guess'], n_engines_failed=0, highlift_config='landing', lg_down=1, h_ground=10.668)

print('LANDING: \n' + pprint.pformat(dragDict_landing))
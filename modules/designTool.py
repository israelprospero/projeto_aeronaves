'''
Conceptual Aircraft Design Tool
(for PRJ-22 and AP-701 courses)

Maj. Eng. Ney Rafael Secco (ney@ita.br)
Aircraft Design Department
Aeronautics Institute of Technology

05-2025

The code uses several historical regression from
aircraft design books to make a quick initial
sizing procedure.

Generally, the user should call only the 'analyze'
function from this module.
'''

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CONSTANTS
ft2m = 0.3048
kt2ms = 0.514444
lb2N = 4.44822
nm2m = 1852.0
gravity = 9.81
gamma_air = 1.4
R_air = 287

#========================================
# MAIN FUNCTION

def analyze(airplane = None,
            print_log = False, # Plot results on the terminal screen
            plot = False, # Generate 3D plot of the aircraft
            ):
    '''
    This is the main function thadeft should be used for aircraft analysis.
    '''

    # Load standard airplane if none is provided
    if airplane is None:
        airplane = standard_airplane()

    # Use an average wing loading for transports
    # to estime W0_guess and T0_guess if none are provided
    if 'W0_guess' in airplane.keys():
        W0_guess = airplane['W0_guess']
    else:
        W0_guess = 5e3*airplane['S_w']
    
    if 'T0_guess' in airplane.keys():
        T0_guess = airplane['T0_guess']
    else:
        T0_guess = 0.3*W0_guess

    # Generate geometry
    geometry(airplane)

    if plot:
        plot3d(airplane)

    # Converge MTOW and Takeoff Thrust
    thrust_matching(W0_guess, T0_guess, airplane)

    # Balance analysis
    balance(airplane)

    # Landing gear design
    landing_gear(airplane)

    # Direct operating cost
    doc(airplane, plot=plot)


    if print_log:
        print('W_empty [kgf]: %d'%(airplane['W_empty']/gravity))
        print('W_fuel [kgf]: %d'%(airplane['W_fuel']/gravity))
        print('W0 [kgf]: %d'%(airplane['W0']/gravity))
        print('T0 [kgf]: %d'%(airplane['T0']/gravity))
        print('T0/W0: %.3f'%(airplane['T0']/airplane['W0']))
        print('W0/S [kgf/m2]: %d'%(airplane['W0']/gravity/airplane['S_w']))
        print('deltaS_wlan [m2]: %.1f'%(airplane['deltaS_wlan']))
        print('tank_excess [%%]: %.1f'%(airplane['tank_excess']*100))
        print('V_maxfuel [L]: %d'%(airplane['V_maxfuel']*1000))
        print('CLv: %.3f'%(airplane['CLv']))
        print('SM_fwd [%%]: %.1f'%(airplane['SM_fwd']*100))
        print('SM_aft [%%]: %.1f'%(airplane['SM_aft']*100))
        print('xnp [%%MAC]: %.1f'%((airplane['xnp']-airplane['xm_w'])/airplane['cm_w']*100))
        print('xcg_fwd [%%MAC]: %.1f'%((airplane['xcg_fwd']-airplane['xm_w'])/airplane['cm_w']*100))
        print('xcg_aft [%%MAC]: %.1f'%((airplane['xcg_aft']-airplane['xm_w'])/airplane['cm_w']*100))
        
        if airplane['frac_nlg_fwd'] is not None:
            print('x_mlg [%%MAC]: %.1f'%((airplane['x_mlg']-airplane['xm_w'])/airplane['cm_w']*100))
            print('frac_nlg_fwd [%%]: %.1f'%(airplane['frac_nlg_fwd']*100))
            print('frac_nlg_aft [%%]: %.1f'%(airplane['frac_nlg_aft']*100))
            print('alpha_tipback [deg]: %.1f'%(airplane['alpha_tipback']*180.0/np.pi))
            print('alpha_tailstrike [deg]: %.1f'%(airplane['alpha_tailstrike']*180.0/np.pi))
            print('phi_overturn [deg]: %.1f'%(airplane['phi_overturn']*180.0/np.pi))

        if 'DOC' in airplane.keys():
            print('DOC [$/nm]: %.2f'%airplane['DOC'])
            #airplane['DOC_breakdown'] = DOC_breakdown

    # Plot again now that we have CG and NP
    if plot:
        plot3d(airplane)

    return airplane

#========================================
# DISCIPLINE MODULES

def geometry(airplane):

    # Unpack dictionary
    S_w = airplane['S_w']
    AR_w = airplane['AR_w']
    taper_w = airplane['taper_w']
    sweep_w = airplane['sweep_w']
    dihedral_w = airplane['dihedral_w']
    xr_w = airplane['xr_w']
    zr_w = airplane['zr_w']
    Cht = airplane['Cht']
    AR_h = airplane['AR_h']
    taper_h = airplane['taper_h']
    sweep_h = airplane['sweep_h']
    dihedral_h = airplane['dihedral_h']
    Lc_h = airplane['Lc_h']
    zr_h = airplane['zr_h']
    Cvt = airplane['Cvt']
    AR_v = airplane['AR_v']
    taper_v = airplane['taper_v']
    sweep_v = airplane['sweep_v']
    Lb_v = airplane['Lb_v']
    zr_v = airplane['zr_v']

    ### ADD CODE FROM SECTION 3.1 HERE ###
    
    # 1. Wing span
    b_w = np.sqrt(AR_w * S_w)

    # 2. Root chord
    cr_w = (2 * S_w) / (b_w * (1 + taper_w))

    # 3. Tip chord
    ct_w = taper_w * cr_w

    # 4. Wing tip y-position
    yt_w = b_w / 2

    # 5. Wing tip x-position
    xt_w = xr_w + yt_w * np.tan(sweep_w) + (cr_w - ct_w) / 4

    # 6. Wing tip z-position
    zt_w = zr_w + yt_w * np.tan(dihedral_w)
        
    # 7. Mean aerodynamic chord of the wing
    cm_w = (2 * cr_w / 3) * ((1 + taper_w + taper_w**2) / (1 + taper_w))

    # 8. Lateral position of the mean aerodynamic chord
    ym_w = (b_w / 6) * ((1 + 2 * taper_w) / (1 + taper_w))

    # 9. Longitudinal position of the mean aerodynamic chord
    xm_w = xr_w + ym_w * np.tan(sweep_w) + (cr_w - cm_w) / 4

    # 10. Vertical position of the mean aerodynamic chord
    zm_w = zr_w + ym_w * np.tan(dihedral_w)

    # 11. Horizontal tail moment arm
    Lh = Lc_h * cm_w

    # 12. Horizontal tail area
    S_h = (S_w * cm_w * Cht) / Lh
    
    b_h = np.sqrt(AR_h * S_h)
    
    cr_h = 2*S_h/(b_h*(1 + taper_h))
    
    ct_h = taper_h * cr_h
    
    cm_h = 2*cr_h*(1 + taper_h + taper_h**2)/(3 * (1 + taper_h))
    
    xm_h = xm_w + Lh + (cm_w - cm_h)/4
    
    ym_h = b_h * (1 + 2*taper_h)/(6 * (1 + taper_h))
    
    zm_h = zr_h + ym_h * np.tan(dihedral_h)
    
    xr_h = xm_h - ym_h * np.tan(sweep_h) + (cm_h - cr_h)/4
    
    yt_h = b_h/2
    
    xt_h = xr_h + yt_h * np.tan(sweep_h) + (cr_h - ct_h)/4
    
    zt_h = zr_h + yt_h*np.tan(dihedral_h)
    
    L_v = Lb_v * b_w
    
    S_v = S_w*b_w * Cvt / L_v
    
    b_v = np.sqrt(AR_v * S_v)
    
    cr_v = 2*S_v/(b_v * (1 + taper_v))
    
    ct_v = taper_v * cr_v
    
    cm_v = 2*cr_v * (1 + taper_v + taper_v**2)/(3 * (1 + taper_v))
    
    xm_v = xm_w + L_v + (cm_w - cm_v)/4
    
    zm_v = zr_v + b_v * (1 + 2 * taper_v)/(3 * (1 + taper_v))
    
    xr_v = xm_v - (zm_v - zr_v)*np.tan(sweep_v) + (cm_v - cr_v)/4
    
    zt_v = zr_v + b_v
    
    xt_v = xr_v + (zt_v - zr_v)*np.tan(sweep_v) + (cr_v - ct_v)/4


    # Update dictionary with new results
    airplane['b_w'] = b_w
    airplane['cr_w'] = cr_w
    airplane['xt_w'] = xt_w
    airplane['yt_w'] = yt_w
    airplane['zt_w'] = zt_w
    airplane['ct_w'] = ct_w
    airplane['xm_w'] = xm_w
    airplane['ym_w'] = ym_w
    airplane['zm_w'] = zm_w
    airplane['cm_w'] = cm_w
    airplane['S_h'] = S_h
    airplane['b_h'] = b_h
    airplane['xr_h'] = xr_h
    airplane['cr_h'] = cr_h
    airplane['xt_h'] = xt_h
    airplane['yt_h'] = yt_h
    airplane['zt_h'] = zt_h
    airplane['ct_h'] = ct_h
    airplane['xm_h'] = xm_h
    airplane['ym_h'] = ym_h
    airplane['zm_h'] = zm_h
    airplane['cm_h'] = cm_h
    airplane['S_v'] = S_v
    airplane['b_v'] = b_v
    airplane['xr_v'] = xr_v
    airplane['cr_v'] = cr_v
    airplane['xt_v'] = xt_v
    airplane['zt_v'] = zt_v
    airplane['ct_v'] = ct_v
    airplane['xm_v'] = xm_v
    airplane['zm_v'] = zm_v
    airplane['cm_v'] = cm_v    

    # All variables are stored in the dictionary.
    # There is no need to return anything
    return None

#----------------------------------------

def aerodynamics(airplane, Mach, altitude, CL, W0_guess,
                 n_engines_failed=0, highlift_config='clean',
                 lg_down=0, h_ground=0, method=2,
                 ind_drag_method='Nita',
                 ind_drag_flap_method='Roskam'):
    '''
    Mach: float -> Freestream Mach number.
    
    altitude: float -> Flight altitude [meters].
    
    CL: float -> Lift coefficient

    W0_guess: float -> Latest MTOW estimate [N]
    
    n_engines_failed: integer -> number of engines failed. Windmilling drag is
                                 added here. This number should be less than the
                                 total number of engines.
    
    highlift_config: 'clean', 'takeoff', or 'landing' -> Configuration of high-lift devices
    
    lg_down: 0 or 1 -> 0 for retraced landing gear or 1 for extended landing gear
    
    h_ground: float -> Distance between wing and the ground for ground effect [m].
                       Use 0 for no ground effect.
    
    method: 1 or 2 -> Method 1 applies a single friction coefficient
                      to the entire wetted area of the aircraft (based on Howe).
                      Method 2 is more refined since it computes friction and
                      form factors for each component.

    ind_drag_flap_method: 'Roskam' or 'Raymer' -> Roskam is simpler for conceptual design.
    '''

    # Unpacking dictionary
    S_w = airplane['S_w']
    AR_w = airplane['AR_w']
    cr_w = airplane['cr_w']
    ct_w = airplane['ct_w']
    taper_w = airplane['taper_w']
    sweep_w = airplane['sweep_w']
    tcr_w = airplane['tcr_w']
    tct_w = airplane['tct_w']
    b_w = airplane['b_w']
    cm_w = airplane['cm_w']
    
    clmax_w = airplane['clmax_w']
    k_korn = airplane['k_korn']

    S_h = airplane['S_h']
    cr_h = airplane['cr_h']
    ct_h = airplane['ct_h']
    taper_h = airplane['taper_h']
    sweep_h = airplane['sweep_h']
    tcr_h = airplane['tcr_h']
    tct_h = airplane['tct_h']
    b_h = airplane['b_h']
    cm_h = airplane['cm_h']
    
    S_v = airplane['S_v']
    cr_v = airplane['cr_v']
    ct_v = airplane['ct_v']
    taper_v = airplane['taper_v']
    sweep_v = airplane['sweep_v']
    tcr_v = airplane['tcr_v']
    tct_v = airplane['tct_v']
    b_v = airplane['b_v']
    cm_v = airplane['cm_v']
    
    L_f = airplane['L_f']
    D_f = airplane['D_f']
    
    L_n = airplane['L_n']
    D_n = airplane['D_n']
    
    x_nlg = airplane['x_nlg'] # This is only used to check if we have LG
    
    n_engines = airplane['n_engines']
    n_engines_under_wing = airplane['n_engines_under_wing']

    flap_type = airplane['flap_type']
    c_flap_c_wing = airplane['c_flap_c_wing']
    b_flap_b_wing = airplane['b_flap_b_wing']
    
    slat_type = airplane['slat_type']
    c_slat_c_wing = airplane['c_slat_c_wing']
    b_slat_b_wing = airplane['b_slat_b_wing']
    
    k_exc_drag = airplane['k_exc_drag']

    has_winglet = airplane['winglet']
    
    # Default rugosity value (smooth paint from Raymer Tab 12.5)
    rugosity = 0.634e-5
    
    # WING
    Shid__S_w = D_f/(b_w * (1 + taper_w)) * (2 - D_f/b_w * (1 - taper_w))
    Sexp_w = (1 - Shid__S_w)*S_w
    Swet_w = 2 * Sexp_w * (1 + tcr_w/(4 * (1 + taper_w)) * (1 + taper_w * tcr_w/tct_w))    
    
    # If the aircraft has winglet
    if has_winglet:
        taper_winglet = 0.21
        tc_winglet = 0.08
        Swet_w = Swet_w + 2 * ct_w**2 * (1 + taper_winglet) * (1 + tc_winglet/(4*(1 + taper_winglet)) * (1 + taper_winglet))
    
    Cf_w = Cf_calc(Mach=Mach, altitude=altitude, length = cm_w, rugosity = rugosity, k_lam = 0.05)
    FF_w = FF_surface(Mach, tcr= tcr_w, tct= tct_w, sweep= sweep_w, b= b_w, cr= cr_w, ct= ct_w)    
    Qw = 1
    CD0_w = Cf_w * FF_w * Qw * Swet_w/S_w
    
    # HORIZONTAL TAIL
    Sexp_h = S_h
    Swet_h = 2 * Sexp_h * (1 + tcr_h/(4 * (1 + taper_h)) * (1 + taper_h * tcr_h/tct_h))    
    Cf_h = Cf_calc(Mach, altitude, length = cm_h, rugosity = rugosity, k_lam = 0.05)
    FF_h = FF_surface(Mach, tcr_h, tct_h, sweep_h, b_h, cr_h, ct_h)    
    Qh = 1.04
    CD0_h = Cf_h * FF_h * Qh * Swet_h/S_w
    
    # VERTICAL TAIL
    Sexp_v = S_v
    Swet_v = 2 * Sexp_v * (1 + tcr_v/(4 * (1 + taper_v)) * (1 + taper_v * tcr_v/tct_v))
    Cf_v = Cf_calc(Mach, altitude, length = cm_v, rugosity = rugosity, k_lam = 0.05)
    FF_v = FF_surface(Mach, tcr_v, tct_v, sweep_v, 2*b_v, cr_v, ct_v)
    Qv = 1.04
    CD0_v = Cf_v * FF_v * Qv * Swet_v/S_w
    
    # FUSELAGE
    fitness_f = L_f/D_f
    Swet_f = np.pi * D_f * L_f * (1 - 2/fitness_f)**(2/3) * (1 + 1/(fitness_f**2))
    Cf_f = Cf_calc(Mach, altitude, length = L_f, rugosity = rugosity, k_lam = 0.05)
    FF_f = 1 + 60/(fitness_f**3) + fitness_f/400
    Qf = 1
    CD0_f = Cf_f*FF_f*Qf*Swet_f/S_w
    
    # NACELLES
    Swet_n = n_engines * np.pi * D_n * L_n
    Cf_n = Cf_calc(Mach, altitude, length = L_n, rugosity = rugosity, k_lam = 0.05)
    FF_n = 1 + 0.35*D_n/L_n
    Qn = 1.2
    CD0_n = Cf_n * FF_n * Qn * Swet_n/S_w
    
    # Clean Configuration
    CD0_clean = CD0_w + CD0_h + CD0_v + CD0_f + CD0_n
    
    # IF THERE'S NO WINGLET
    AR_eff = AR_w
    
    ## IF THERE'S WINGLET
    # AR_eff = 1.2*AR_w
    
    Delta_taper = -0.357 + 0.45 * np.exp(-0.0375 * 180 * sweep_w / np.pi)
    taper_opt = taper_w - Delta_taper
    f_taper = 0.0524 * taper_opt**4 - 0.15 * taper_opt**3 + 0.1659*taper_opt**2 - 0.0706*taper_opt + 0.0119
    e_theo = 1/(1 + f_taper * AR_eff)
    
    k_em = 1/(1 + 0.12 * Mach**6)
    k_ef = 1 - 2*(D_f / b_w)**2
    
    e_clean = e_theo * k_ef * k_em * 0.873 # commercial jets only
    
    # Wave Drag
    tc_avg_w = 0.25*tcr_w + 0.75*tct_w
    sweep_50 = geo_change_sweep(0.25, 0.50, sweep_w, b_w/2, cr_w, ct_w)
    Mdd = k_korn/np.cos(sweep_50) - tc_avg_w/((np.cos(sweep_50)**2)) - CL/(10*(np.cos(sweep_50))**3)
    Mc = Mdd - (0.1/80)**(1/3)
    CD_wave = 20*(max(0, Mach - Mc))**4
    
    # Maximun lift coefficient and high lift devices
    CL_max_clean = 0.9 * clmax_w * np.cos(sweep_w)
    
    if flap_type is not None:
        Sflap__S_w = b_flap_b_wing * (2 - b_flap_b_wing * (1 - taper_w))/(1 + taper_w) - Shid__S_w
    
    if slat_type is not None:
        Sslat__S_w = b_slat_b_wing * (2 - b_slat_b_wing * (1 - taper_w))/(1 + taper_w) - Shid__S_w
        
    sweep_flap = geo_change_sweep(0.25, 1-c_flap_c_wing, sweep_w, b_w/2, cr_w, ct_w)
    
    if flap_type == 'plain':
        delta_cl_max_flap = 0.9
        F_flap = 0.9
        
    elif flap_type == 'single slotted':
        delta_cl_max_flap = 1.3 * (1 + c_flap_c_wing)
        F_flap = 1
        
    elif flap_type == 'double slotted':
        delta_cl_max_flap = 1.6 * (1 + c_flap_c_wing)
        F_flap = 1.2
        
    elif flap_type == 'triple slotted':
        delta_cl_max_flap = 1.9 * (1 + c_flap_c_wing)
        F_flap = 1.5
        
    if highlift_config == 'clean':
        delta_lift = 0
    elif highlift_config == 'takeoff':
        delta_lift = 0.75
    elif highlift_config == 'landing':
        delta_lift = 1
    
    Delta_CL_max_flap = 0.9 * delta_cl_max_flap * Sflap__S_w * np.cos(sweep_flap) * delta_lift * c_flap_c_wing/0.3
    
    if highlift_config == 'clean':
        CD0_flap = 0
        Delta_e_flap = 0
    elif highlift_config == 'takeoff':
        CD0_flap = (0.03 * F_flap - 0.004)/(AR_eff**0.33)
        Delta_e_flap = -0.05
    elif highlift_config == 'landing':
        CD0_flap = (0.12 * F_flap)/(AR_eff**0.33)
        Delta_e_flap = -0.1
        
    # Leading Edge devices (slats)
    if slat_type is not None:
        sweep_slat = geo_change_sweep(0.25, c_slat_c_wing, sweep_w, b_w/2, cr_w, ct_w)
        
        if slat_type == 'slot':
            delta_cl_max_slat = 0.2
            
        elif slat_type == 'leading edge flap':
            delta_cl_max_slat = 0.3
            
        elif slat_type == 'Kruger flap':
            delta_cl_max_slat = 0.3
            
        elif slat_type == 'moving slat':
            delta_cl_max_slat = 0.4 * (1 + c_flap_c_wing)
        
        Delta_CL_max_slat = 0.9 * delta_cl_max_slat * Sslat__S_w * np.cos(sweep_slat) * delta_lift * c_slat_c_wing/0.15
    
        CD0_slat = CD0_w * c_slat_c_wing * Sslat__S_w * np.cos(sweep_w) * delta_lift
        
    else:
        Delta_CL_max_slat = 0
        CD0_slat = 0
        
    
    CLmax = CL_max_clean + Delta_CL_max_flap + Delta_CL_max_slat
    
    # Induced Drag    
    e = e_clean + Delta_e_flap
    K = 1/(np.pi * AR_eff * e)
    
    if h_ground > 0:
        GE = 33 * (h_ground/b_w)**1.5
        K_GE = GE/(1 + GE)
        K = K * K_GE
    
    CD_ind = K * CL**2
    
    #K_clean = 1/(np.pi * AR_eff * e_clean)
    #CD_ind_clean = K_clean * CL**2
    #
    #if Delta_e_flap != 0:
    #    K_flap = 1/(np.pi * AR_eff * Delta_e_flap)
    #    CD_ind_flap = K_flap * CL**2
    #else:
    #    CD_ind_flap = 0
    
    # Additional components
    if lg_down == 1:
        CD0_lg = 0.02
    else:
        CD0_lg = 0
    
    if n_engines_failed > 0:
        CD0_windmill = n_engines_failed * 0.3 * np.pi/4 * D_n**2 / S_w
    else:
        CD0_windmill = 0
        
    # Excrescence drag and total parasite drag
    CD0 = (CD0_clean + CD0_flap + CD0_slat + CD0_lg + CD0_windmill)/(1 - k_exc_drag)
    CD0_exc = CD0 * k_exc_drag
    
    # Total Drag
    CD = CD0 + CD_ind + CD_wave

    Swet = Swet_f + Swet_h + Swet_v + Swet_n + Swet_w

    # Create a drag breakdown dictionary
    dragDict = {'CD': CD,
                'CD0_flap' : CD0_flap,
                'CD0_slat' : CD0_slat,
                'CD0_lg' : CD0_lg,
                'CD0_wdm' : CD0_windmill, #CD0_wdm,
                'CD0_exc' : CD0_exc,
                'CD0' : CD0,
                'CDind' : CD_ind,
                'CDwave' : CD_wave,
                'CLmax_clean' : CL_max_clean,
                'deltaCLmax_flap' : Delta_CL_max_flap,
                'deltaCLmax_slat' : Delta_CL_max_slat,
                'CLmax' : CLmax,
                'K' : K,
                'e' : e,
                'Swet' : Swet
                #'CDind_clean' : CD_ind_clean, # CDind_clean,
                #'CDind_flap' : CD_ind_flap,
    }

    if method == 2:
        dragDict['CD0_w'] = CD0_w
        dragDict['CD0_h'] = CD0_h
        dragDict['CD0_v'] = CD0_v
        dragDict['CD0_f'] = CD0_f
        dragDict['CD0_n'] = CD0_n

    # Update dictionary
    airplane['Swet_f'] = Swet_f
    airplane['AR_eff'] = AR_eff

    return CD, CLmax, dragDict

#----------------------------------------

def engineTSFC(Mach, altitude, airplane):
    '''
    This function computes the engine thrust-specific fuel
    consumption and thrust correction factor compared to
    static sea-level conditions. The user has to define the
    engine parameters in a 'engine' dictionary within
    the airplane dictionary. The engine model must be
    identified by the 'model' field of the engine dictionary.
    The following engine models are available:
    
    Howe TSFC turbofan model:
    requires the bypass ratio. An optional sea-level TSFC
    could also be provided. Otherwise, standard parameters
    are used.
    airplane['engine'] = {'model': 'howe turbofan',
                          'BPR': 3.04,
                          'Cbase': 0.7/3600} # Could also be None
                          
    Thermodynamic cycle turbojet:
    This model uses a simplified thermodynamic model of
    turbofans to estimate maximum thrust and TSFC
    
    airplane['engine'] = {'model': 'thermo turbojet'
                          'data': dictionary (check turbojet_model function)}
    
    The user can also leave a 'weight' field in the dictionary
    to replace the weight estimation.
    '''
    
    BPR = airplane['engine']['BPR']
    Cbase = airplane['engine']['Cbase']
    
    _,_,rho,_ = atmosphere(altitude, 288.15)
    sigma = rho/1.225

    
    if BPR < 4 :
        Cbase = 0.85/3600
    else:
        Cbase = 0.7/3600
    
    
    C = Cbase*(1 - 0.15*BPR**0.65) * (1 + 0.28 * (1 + 0.063*BPR**2) * Mach) * sigma**(0.08)
    
    kT = (0.0013*BPR - 0.0397) * altitude/1000 - 0.0248*BPR + 0.7125

    return C, kT

#----------------------------------------

def empty_weight(W0_guess, T0_guess, airplane):

    # Unpack dictionary
    S_w = airplane['S_w']
    AR_eff = airplane['AR_eff']
    b_w = airplane['b_w']
    taper_w = airplane['taper_w']
    sweep_w = airplane['sweep_w']
    xm_w = airplane['xm_w']
    cm_w = airplane['cm_w']
    tcr_w = airplane['tcr_w']
    
    S_h = airplane['S_h']
    xm_h = airplane['xm_h']
    cm_h = airplane['cm_h']
    
    S_v = airplane['S_v']
    xm_v = airplane['xm_v']
    cm_v = airplane['cm_v']
    
    L_f = airplane['L_f']
    D_f = airplane['D_f']
    Swet_f = airplane['Swet_f']
    
    n_engines = airplane['n_engines']
    x_n = airplane['x_n']
    L_n = airplane['L_n']
    
    x_nlg = airplane['x_nlg']
    x_mlg = airplane['x_mlg']
    
    flap_type = airplane['flap_type']
    c_flap_c_wing = airplane['c_flap_c_wing']
    b_flap_b_wing = airplane['b_flap_b_wing']
    slat_type = airplane['slat_type']
    c_slat_c_wing = airplane['c_slat_c_wing']
    b_slat_b_wing = airplane['b_slat_b_wing']
    c_ail_c_wing = airplane['c_ail_c_wing']
    b_ail_b_wing = airplane['b_ail_b_wing']
    
    altitude_cruise = airplane['altitude_cruise']
    Mach_cruise = airplane['Mach_cruise']
    
    airplane_type = airplane['type']
    
    
    # Wing Weight calculations
    
    Nz = 1.5*2.5 # Ultimate load factor
    
    # Flaps Area
    if airplane['flap_type'] == 'plain':
        m_flap = 1
    elif airplane['flap_type'] == 'single slotted':
        m_flap = 1.15*1.25
    elif airplane['flap_type'] == 'double slotted':
        m_flap = 1.30*1.25
    elif airplane['flap_type'] == 'triple slotted':
        m_flap = 1.45*1.25     
        
    # S_____S_wing = x1           /(1 + taper_w) * (y2            * (2 - y2           *(1 - taper_w)) - y1      * (2 - y1      * (1 - taper_w))) * m    
    S_flap__S_wing = c_flap_c_wing/(1 + taper_w) * (b_flap_b_wing * (2 - b_flap_b_wing*(1 - taper_w)) - D_f/b_w * (2 - D_f/b_w * (1 - taper_w))) * m_flap
    
    
    # Slats Area
    if airplane['slat_type'] == None:
        m_slat = 0
    elif airplane['slat_type'] == 'leading edge flap':
        m_slat = 1
    elif airplane['slat_type'] == 'Kruger flaps':
        m_slat = 1
    elif airplane['slat_type'] == 'slats':
        m_slat = 1.25   
    # S_____S_wing = x1           /(1 + taper_w) * (y2            * (2 - y2           *(1 - taper_w)) - y1      * (2 - y1      * (1 - taper_w))) * m      
    S_slat__S_wing = c_slat_c_wing/(1 + taper_w) * (b_slat_b_wing * (2 - b_slat_b_wing*(1 - taper_w)) - D_f/b_w * (2 - D_f/b_w * (1 - taper_w))) * m_slat    
    
    # Ailerons Area
    m_aileron = 1    
    # S_____S_wing = x1          /(1 + taper_w) * (y2 *(2 - y2* (1 - taper_w)) - y1                 * (2 - y1                 * (1 - taper_w))) * m     
    S_ail__S_wing  = c_ail_c_wing/(1 + taper_w) * (1 * (2 - 1 * (1 - taper_w)) - (1 - b_ail_b_wing) * (2 - (1 - b_ail_b_wing) * (1 - taper_w))) * m_aileron
    
    # Control Surfaces Area
    S_csw = (S_flap__S_wing + S_slat__S_wing + S_ail__S_wing)*S_w
    
    # Units Conversions
    W0_lbs = W0_guess/lb2N # N to lbs
    S_w__ftsq = S_w / ft2m**2 # m^2 to ft^2
    S_csw__ftsq = S_csw / ft2m**2 # m^2 to ft^2
    
    # Wing Weight (in lbs then kg)
    W_w_lbs = 0.0051 * (W0_lbs * Nz)**0.557 * S_w__ftsq**0.649 * AR_eff**0.55 * tcr_w**(-0.4) * (1 + taper_w)**0.1 * np.cos(sweep_w)**(-1) * S_csw__ftsq**0.1
    W_w = W_w_lbs * lb2N # lbs to N
    
    # Wing Center of Gravity Estimation
    x_CG_w = xm_w + 0.4*cm_w
    
    # Horizontal Tail Weight
    W_h = 27 * gravity * S_h
    
    # HT Center of Gravity Estimation
    x_CG_h = xm_h + 0.4*cm_h
    
    # Vertical Tail Weight
    W_v = 27 * gravity * S_v
    
    # VT Center of Gravity Estimation
    x_CG_v = xm_v + 0.4*cm_v
    
    #Fuselage Weight
    W_f = 24 * gravity * Swet_f
    
    # Fuselage Center of Gravity Estimation
    x_CG_f = 0.45*L_f
    
    # Nose Landing Gear Weight 
    W_nlg = 0.15 * 0.043 * W0_guess
    
    # Nose Landing Gear Center of Gravity Estimation
    x_CG_nlg = x_nlg
    
    # Main Landing Gear Weight 
    W_mlg = 0.85 * 0.043 * W0_guess
    
    # Main Landing Gear Center of Gravity Estimation
    x_CG_mlg = x_mlg
    
    
    T_eng_s = T0_guess / n_engines
    
    # Engine Weight (Isolated)
    W_eng_s = 14.7 * gravity * (T_eng_s/1000)**1.1 * np.exp(-0.045*airplane['engine']['BPR'])
    # Engine Weight (Installed) 
    W_eng_installed = 1.3*n_engines*W_eng_s
    
    # Engine Center of Gravity Estimation
    x_CG_eng = x_n + 0.5*L_n
    
    # Remaining Systemas Weight and Center of Gravity Estimations
    W_allelse = 0.17*W0_guess
    x_CG_ae = 0.45*L_f
    
    # Total Empty Weight
    W_empty = W_w + W_h + W_v + W_f + W_nlg + W_mlg + W_eng_installed + W_allelse
    x_CG_empty = (W_w*x_CG_w + W_h*x_CG_h + W_v*x_CG_v + W_f*x_CG_f + W_nlg*x_CG_nlg + W_mlg*x_CG_mlg + W_eng_installed*x_CG_eng + W_allelse*x_CG_ae)/W_empty

    # Update dictionary
    airplane['W_w'] = W_w
    airplane['W_h'] = W_h
    airplane['W_v'] = W_v
    airplane['W_f'] = W_f
    airplane['W_nlg'] = W_nlg
    airplane['W_mlg'] = W_mlg
    airplane['W_eng'] = W_eng_installed
    airplane['W_allelse'] = W_allelse
    airplane['xcg_empty'] = x_CG_empty

    return W_empty

#----------------------------------------

def fuel_weight(W0_guess, airplane, range_cruise, update_Mf_hist=False):

    # Unpacking dictionary
    S_w = airplane['S_w']
    
    altitude_cruise = airplane['altitude_cruise']
    Mach_cruise = airplane['Mach_cruise']
    loiter_time = airplane['loiter_time']
    
    altitude_altcruise = airplane['altitude_altcruise']
    Mach_altcruise = airplane['Mach_altcruise']
    range_altcruise = airplane['range_altcruise']
    
    airplane_type = airplane['type']
    
    # TSFC computation
    C_cruise, _ = engineTSFC(Mach_cruise, altitude_cruise, airplane)
    
    # Mission mass fractions (from Roskam)
    Mf_start = 0.990
    Mf_taxi = 0.990
    Mf_takeoff = 0.995
    Mf_climb = 0.980
    Mf_descent = 0.990
    Mf_landing = 0.992

    # Cruise
    W_cruise = W0_guess * Mf_start * Mf_taxi * Mf_takeoff * Mf_climb
    T, p, rho, mi = atmosphere(altitude_cruise, 288.15)
    a_cruise = np.sqrt(gamma_air * R_air * T)
    V_cruise = Mach_cruise * a_cruise
    CL_cruise = 2 * W_cruise / (rho * S_w * V_cruise**2)
    CD_cruise, _, dragDict_cruise = aerodynamics(airplane, Mach_cruise, altitude_cruise, CL_cruise, W0_guess)
    Mf_cruise = np.exp(-range_cruise * C_cruise * CD_cruise / (V_cruise * CL_cruise))

    # Loiter
    CD0 = dragDict_cruise['CD0']
    CDwave = dragDict_cruise['CDwave']
    K = dragDict_cruise['K']
    L_D_max = 1 / (2 * np.sqrt((CD0 + CDwave) * K))
    C_loiter = C_cruise - 0.1 / 3600
    Mf_loiter = np.exp(-loiter_time * C_loiter / L_D_max)

    # Alt cruise
    C_altcruise, _ = engineTSFC(Mach_altcruise, altitude_altcruise, airplane)
    T_alt, p_alt, rho_alt, mi_alt = atmosphere(altitude_altcruise, 288.15)
    a_altcruise = np.sqrt(gamma_air * R_air * T_alt)
    V_altcruise = Mach_altcruise * a_altcruise
    W_altcruise = W_cruise * Mf_cruise * Mf_loiter * Mf_descent
    CL_alt = 2 * W_altcruise / (rho_alt * S_w * V_altcruise**2)
    CD_alt, _, _ = aerodynamics(airplane, Mach_altcruise, altitude_altcruise, CL_alt, W0_guess)
    Mf_altcruise = np.exp(-range_altcruise * C_altcruise * CD_alt / (V_altcruise * CL_alt))

    # Total Mf for consumed fuel
    Mf = Mf_start * Mf_taxi * Mf_takeoff * Mf_climb * Mf_cruise * Mf_loiter * Mf_descent * Mf_altcruise * Mf_landing
    
    kf = 1.06  # trapped fuel correction
    W_fuel = kf * (1 - Mf) * W0_guess  # total fuel (used + trapped)
    
    W_used_fuel = W_fuel / kf
    W_trapped_fuel = W_fuel - W_used_fuel
    Mf_trapped = 1 - W_trapped_fuel / W0_guess

    # Store Mfs
    airplane.update({
        'Mf_engine_start': Mf_start,
        'Mf_taxi': Mf_taxi,
        'Mf_takeoff': Mf_takeoff,
        'Mf_climb': Mf_climb,
        'Mf_cruise': Mf_cruise,
        'Mf_loiter': Mf_loiter,
        'Mf_descent': Mf_descent,
        'Mf_altcruise': Mf_altcruise,
        'Mf_landing': Mf_landing,
        'Mf_total': Mf,
        'Mf_trapped': Mf_trapped,
        'fuel_trapped': W_trapped_fuel / gravity
    })

    return W_fuel, Mf_cruise, CL_cruise, CD_cruise, C_cruise, L_D_max, C_loiter, CL_alt, CD_alt, C_altcruise

#----------------------------------------

def weight(W0_guess, T0_guess, airplane):
    W_payload = airplane['W_payload']
    W_crew = airplane['W_crew']
    range_cruise = airplane['range_cruise']
    delta = 1000

    while abs(delta) > 10:
        W_fuel, Mf_cruise, _, _, _, _, _, _, _, _ = fuel_weight(W0_guess, airplane, range_cruise)
        W_empty = empty_weight(W0_guess, T0_guess, airplane)
        W0 = W_empty + W_fuel + W_payload + W_crew
        delta = W0 - W0_guess
        W0_guess = W0

    airplane['W0'] = W0
    airplane['W_empty'] = W_empty
    airplane['W_fuel'] = W_fuel

    # Phases (sem 'trapped' por enquanto)
    phases = ['engine_start', 'taxi', 'takeoff', 'climb', 'cruise',
              'loiter', 'descent', 'altcruise', 'landing']
    
    Mfs = [airplane['Mf_' + p] for p in phases]

    W = W0 / gravity
    Wfuel = W_fuel / gravity

    # Garantir combustível preso em kg
    W_used = W_fuel / 1.06  # usado
    fuel_trapped = Wfuel - (W_used / gravity)
    airplane['fuel_trapped'] = fuel_trapped

    airplane['W_gross_total'] = W
    airplane['W_gross_fuel_total'] = Wfuel

    fuel_breakdown = {}
    percent_breakdown = {}
    mf_breakdown = {}

    for phase, mf in zip(phases, Mfs):
        fuel = W * (1 - mf)
        percent = 100 * fuel / Wfuel if Wfuel > 0 else 0
        airplane[f'W_gross_{phase}'] = W
        fuel_breakdown[phase.replace('_', ' ').title()] = fuel
        percent_breakdown[phase.replace('_', ' ').title()] = percent
        mf_breakdown[phase.replace('_', ' ').title()] = mf
        W *= mf

    # Adiciona "Trapped fuel" ao final
    fuel_breakdown['Trapped fuel'] = fuel_trapped
    percent_breakdown['Trapped fuel'] = 100 * fuel_trapped / Wfuel
    mf_breakdown['Trapped fuel'] = airplane['Mf_trapped']

    airplane['fuel_breakdown'] = fuel_breakdown
    airplane['fuel_percent_breakdown'] = percent_breakdown
    airplane['fuel_Mf_breakdown'] = mf_breakdown
    airplane['fuel_total_used'] = sum(fuel_breakdown[p] for p in fuel_breakdown if p != 'Trapped fuel')
    airplane['fuel_total'] = Wfuel

    return W0, W_empty, W_fuel, Mf_cruise


#----------------------------------------

def performance(W0, Mf_cruise, airplane):

    '''
    This function computes the required thrust and wing areas
    required to meet takeoff, landing, climb, and cruise requirements.

    OUTPUTS:
    T0: real -> Total thrust required to meet all mission phases
    deltaS_wlan: real -> Wing area margin for landing. This value should be positive
                         for a feasible landing.
    '''

    # Unpacking dictionary
    S_w = airplane['S_w']
    
    n_engines = airplane['n_engines']
    
    h_ground = airplane['h_ground']
    
    altitude_takeoff = airplane['altitude_takeoff']
    distance_takeoff = airplane['distance_takeoff']
    
    altitude_landing = airplane['altitude_landing']
    distance_landing = airplane['distance_landing']
    MLW_frac = airplane['MLW_frac']
    
    altitude_cruise = airplane['altitude_cruise']
    
    Mach_cruise = airplane['Mach_cruise']


    # Get the maximum required thrust with a 5% margin
    T0vec = [T0_to, T0_cruise, T0_1, T0_2, T0_3, T0_4, T0_5, T0_6]
    T0 = 1.05*max(T0vec)

    return T0, T0vec, deltaS_wlan, CLmaxTO

#----------------------------------------

def thrust_matching(W0_guess, T0_guess, airplane):

    # Set iterator
    delta = 1000

    # Loop to adjust T0
    while abs(delta) > 10:

        W0, W_empty, W_fuel, Mf_cruise = weight(W0_guess, T0_guess, airplane)

        T0, T0vec, deltaS_wlan, CLmaxTO = performance(W0, Mf_cruise, airplane)

        # Compute change with respect to previous iteration
        delta = T0 - T0_guess

        # Update guesses for the next iteration
        T0_guess = T0
        W0_guess = W0

    # Update dictionary
    airplane['W0'] = W0
    airplane['W_empty'] = W_empty
    airplane['W_fuel'] = W_fuel
    airplane['T0'] = T0
    airplane['T0vec'] = T0vec
    airplane['deltaS_wlan'] = deltaS_wlan
    airplane['CLmaxTO'] = CLmaxTO

    # Return
    return None

#----------------------------------------

def balance(airplane):

    # Unpack dictionary
    W0 = airplane['W0']
    W_payload = airplane['W_payload']
    xcg_payload = airplane['xcg_payload']
    W_crew = airplane['W_crew']
    xcg_crew = airplane['xcg_crew']
    W_empty = airplane['W_empty']
    xcg_empty = airplane['xcg_empty']
    W_fuel = airplane['W_fuel']
    
    Mach_cruise = airplane['Mach_cruise']
    
    S_w = airplane['S_w']
    AR_eff = airplane['AR_eff']
    taper_w = airplane['taper_w']
    sweep_w = airplane['sweep_w']
    b_w = airplane['b_w']
    xr_w = airplane['xr_w']
    zr_w = airplane['zr_w']
    cr_w = airplane['cr_w']
    ct_w = airplane['ct_w']
    xm_w = airplane['xm_w']
    cm_w = airplane['cm_w']
    tcr_w = airplane['tcr_w']
    tct_w = airplane['tct_w']
    
    S_h = airplane['S_h']
    AR_h = airplane['AR_h']
    sweep_h = airplane['sweep_h']
    b_h = airplane['b_h']
    cr_h = airplane['cr_h']
    ct_h = airplane['ct_h']
    xm_h = airplane['xm_h']
    zm_h = airplane['zm_h']
    cm_h = airplane['cm_h']
    eta_h = airplane['eta_h']
    Lc_h = airplane['Lc_h']
    
    Cvt = airplane['Cvt']
    
    L_f = airplane['L_f']
    D_f = airplane['D_f']
    
    y_n = airplane['y_n']
    
    T0 = airplane['T0']
    n_engines = airplane['n_engines']

    c_tank_c_w = airplane['c_tank_c_w']
    x_tank_c_w = airplane['x_tank_c_w']
    b_tank_b_w_start = airplane['b_tank_b_w_start']
    b_tank_b_w_end = airplane['b_tank_b_w_end']

    rho_fuel = airplane['rho_fuel']
    
    CLmaxTO = airplane['CLmaxTO']


    # Update dictionary
    airplane['xcg_fwd'] = xcg_fwd
    airplane['xcg_aft'] = xcg_aft
    airplane['xnp'] = xnp
    airplane['SM_fwd'] = SM_fwd
    airplane['SM_aft'] = SM_aft
    airplane['tank_excess'] = tank_excess
    airplane['V_maxfuel'] = V_maxfuel
    airplane['CLv'] = CLv

    return None

#----------------------------------------

def landing_gear(airplane):

    # Unpack dictionary
    x_nlg = airplane['x_nlg']
    x_mlg = airplane['x_mlg']
    y_mlg = airplane['y_mlg']
    z_lg = airplane['z_lg']
    xcg_fwd = airplane['xcg_fwd']
    xcg_aft = airplane['xcg_aft']
    x_tailstrike = airplane['x_tailstrike']
    z_tailstrike = airplane['z_tailstrike']


    # Update dictionary
    airplane['frac_nlg_fwd'] = frac_nlg_fwd
    airplane['frac_nlg_aft'] = frac_nlg_aft
    airplane['alpha_tipback'] = alpha_tipback
    airplane['alpha_tailstrike'] = alpha_tailstrike
    airplane['phi_overturn'] = phi_overturn

    return None

#----------------------------------------

def doc(airplane, CEF=6.0, plot=False):
    # This function computes Direct Operating Cost (DOC) using Roskam's Part VIII methodology.
    # CEF is the cost estimation factor for the current year, taking 1989 as reference (CEF = 1.0)

    # Unpack dictionary
    Rbl = airplane['block_range']
    tbl = airplane['block_time']
    W0 = airplane['W0']
    W_empty = airplane['W_empty']
    Weng = airplane['W_eng']
    T0 = airplane['T0']
    n_eng = airplane['n_engines']
    n_captains = airplane['n_captains']
    n_copilots = airplane['n_copilots']
    rho_fuel = airplane['rho_fuel']

    # Estimate block fuel
    W_fuel, _ = fuel_weight(W0, airplane, range_cruise=Rbl)
    
    # UNIT CONVERSIONS
    
    # Block range in NM
    Rbl = Rbl/nm2m

    # Block time [h] (can be estimated using actual flight times between cities with the same range)
    tbl = tbl/3600

    # Fuel weight [lbs]
    Wfbl = W_fuel/lb2N
    
    # Takeoff weight [lbs]
    W0 = W0/lb2N
    
    # Empty weight [lbs]
    W_empty = W_empty/lb2N
    
    # Engine weight [lbs]
    Weng = Weng/lb2N
    
    # Total takeoff thrust (all engines) [lbs]
    T0 = T0/lb2N
    
    # Number of engines
    n_eng = 2

    # Block speed
    Vbl = Rbl / tbl

    # CREW COST

    # ncj represents the number of Captains, Copilots, and Flight Engineers
    ncj = np.array([n_captains, n_copilots, 0])
    Kj = 0.26 # Vacation factor
    
    # Base annual salaries
    SAL89 = np.array([35000., 24000., 20000.])
    
    # Cost estimation factor
    CEF89 = 3.02
    SAL = SAL89 * CEF / CEF89
    
    AH = 800 # Expected flight hours per year

    CEF2019 = 6.0
    TEF = 8.5*CEF/CEF2019 # Travel expenses per crew member (US$/hour)
    
    C_crew = ncj/Vbl * ( ( 1 + Kj ) * SAL / AH + TEF )
    C_crew= C_crew.sum()
    
    # FUEL COST
    
    # Fuel price (USD / Gallon) (based in 2022)
    FP = 3.20

    # Fuel density (lb/gallon)
    FD = rho_fuel*0.0083454
    
    # Fuel cost
    # 1.05 factor is to take into account oil and lubricants
    C_pol = 1.05 * Wfbl / Rbl * FP / FD
    
    # INSURANCE COST
    
    # Insurance factor
    # premium US$ / aircraft US$ / aircraft/ year)
    f_inshull = 0.03  # 0.005<finshull<0.030
    
    # Airplane market price
    AMP89 = 10 ** (3.3191 + 0.8043*np.log10(W0))
    AMP = AMP89 * CEF / CEF89

    # Annual block hours flown by the aircarft
    Uannbl = 1000 * (3.4546 * tbl + 2.994 - np.sqrt(12.289 * tbl ** 2 - 5.6626 * tbl + 8.964))
    
    # Insurance cost
    C_ins = f_inshull * AMP / Uannbl / Vbl
    
    DOCflt = C_crew + C_pol + C_ins
    
    # MAINTENANCE COST
    
    # Airframe maintenance labor cost
    MHRmapbl = 3 + 0.067*(W_empty - Weng)/1000
    Rlap89 = 16.0 # Maintenance labor cost (US$/h in 1989)
    Rlap = Rlap89 * CEF / CEF89
    Clabap = 1.03 * MHRmapbl * Rlap / Vbl
    
    # Engine maintenance labor cost
    Hem = 3000 # Time between overhauls [hours]
    Rleng = Rlap # Same labor cost
    MHRmeng = 0.1 + (0.718 + 0.0317/1000*T0/n_eng)*(1100/Hem)
    Clabeng = 1.03*1.3*n_eng*MHRmeng*Rleng/Vbl
    
    # Airframe maintenance materials
    if W0 < 5000:
        ATF = 0.25
    elif W0 < 10000:
        ATF = 0.50
    else:
        ATF = 1.0
        
    AEP = AMP # Aircraft price
    EP = CEF/CEF89 * 10**(2.3044 + 0.8858*np.log10(T0/n_eng)) # Engine price
    AFP = AEP - n_eng * EP
    Cmatapblhr = 30 * CEF / CEF89 * ATF + 0.79e-5 * AFP
    Cmatap = 1.03*Cmatapblhr / Vbl
    
    # Engine maintenance materials
    ESPPF = 1.5
    Khem = 0.769 + 0.021 * Hem/100
    Cmatengblhr = (5.43e-5 * EP * ESPPF -0.47) / Khem
    Cmateng = 1.03 * 1.3 * n_eng * Cmatengblhr / Vbl
    
    # Additional costs (energy, administration)
    famblab = 1.2
    fambmat = 0.55
    Camb = 1.03/Vbl*(famblab*(MHRmapbl*Rlap+n_eng*MHRmeng*Rleng)+fambmat*(Cmatapblhr + n_eng*Cmatengblhr))
    
    # Maintenance cost
    DOCmnt = Clabap + Clabeng + Cmatap + Cmateng + Camb
    
    # DEPRECIATION COST

    # Depreciation factors (how much value will be lost due to aging)
    Fdap = 0.85
    Fdeng = 0.85
    Fdav = 1
    Fdapsp = 0.85
    Fdengsp = 0.85
    
    # Deprecitation period (how many years until reaching lowest value due to aging)
    DPap = 10
    DPeng = 7
    DPav = 5
    DPapsp = 10
    DPengsp = 7
    
    # Avionics estimated price
    ASP = 0.12*AMP
    
    Cdap = Fdap*(AEP-n_eng*EP-ASP)/DPap/Uannbl/Vbl
    
    Cdeng = Fdeng*n_eng*EP/DPeng/Uannbl/Vbl
    
    Cdav = Fdav*ASP/DPav/Uannbl/Vbl
    
    Fapsp = 0.1
    Cdapsp = Fdapsp*Fapsp*(AEP-n_eng*EP)/DPapsp/Uannbl/Vbl
    
    Fengsp = 0.5
    Cdengsp = Fdengsp*Fengsp*n_eng*EP*ESPPF/DPengsp/Uannbl/Vbl
    
    DOCdepr = Cdap + Cdeng + Cdav + Cdapsp + Cdengsp
    
    
    ## LANDING AND NAVIGATION FEES
    
    # Navigation fee
    Capnf = 10 * CEF/CEF89
    Cnf = Capnf / Vbl/tbl
    
    # Landing fee
    flf = 0.036+4e-8*W0

    # Registration fee
    frf = 0.001 + 1e-8*W0
    
    ## FINANCING DOC
    ffin = 0.07

    # TOTAL DOC
    DOC = (DOCflt + DOCmnt + DOCdepr + Cnf) / (1 - (flf + frf + ffin))
    
    # Recompute relative DOC fractions
    
    # FINANCING DOC
    DOCfin = ffin*DOC

    # LANDING AND NAVIGATION DOC
    Clf = flf*DOC
    Crf = frf*DOC
    DOClanr = Cnf + Clf + Crf

    # Generate dictionary with results
    DOC_breakdown = {'flight':DOCflt,
                     'maintenance':DOCmnt,
                     'depreciation':DOCdepr,
                     'land. and nav. fees':DOClanr,
                     'leasing':DOCfin}

    # Update dictionary
    airplane['DOC'] = DOC
    airplane['DOC_breakdown'] = DOC_breakdown

    if plot:
        
        DOCs = (DOCflt, DOCmnt, DOCdepr, DOClanr, DOCfin)
        labels = ("Flight", "Maintenance", "Depreciation", "Fees", "Leasing")
        fig = plt.figure()
        plt.pie(DOCs,autopct='%1.1f%%', labels=labels)
        plt.show()

    return None

#----------------------------------------
#========================================
# AUXILIARY FUNCTIONS

def atmosphere(z, Tba=288.15):

    '''
    Funçao que retorna a Temperatura, Pressao e Densidade para uma determinada
    altitude z [m]. Essa funçao usa o modelo padrao de atmosfera para a
    temperatura no solo de Tba.
    '''

    # Zbase (so para referencia)
    # 0 11019.1 20063.1 32161.9 47350.1 50396.4

    # DEFINING CONSTANTS
    # Earth radius
    r = 6356766
    # gravity
    g0 = 9.80665
    # air gas constant
    R = 287.05287
    # layer boundaries
    Ht = [0, 11000, 20000, 32000, 47000, 50000]
    # temperature slope in each layer
    A = [-6.5e-3, 0, 1e-3, 2.8e-3, 0]
    # pressure at the base of each layer
    pb = [101325, 22632, 5474.87, 868.014, 110.906]
    # temperature at the base of each layer
    Tstdb = [288.15, 216.65, 216.65, 228.65, 270.65]
    # temperature correction
    Tb = Tba-Tstdb[0]
    # air viscosity
    mi0 = 18.27e-6 # [Pa s]
    T0 = 291.15 # [K]
    C = 120 # [K]

    # geopotential altitude
    H = r*z/(r+z)

    # selecting layer
    if H < Ht[0]:
        raise ValueError('Under sealevel')
    elif H <= Ht[1]:
        i = 0
    elif H <= Ht[2]:
        i = 1
    elif H <= Ht[3]:
        i = 2
    elif H <= Ht[4]:
        i = 3
    elif H <= Ht[5]:
        i = 4
    else:
        raise ValueError('Altitude beyond model boundaries')

    # Calculating temperature
    T = Tstdb[i]+A[i]*(H-Ht[i])+Tb

    # Calculating pressure
    if A[i] == 0:
        p = pb[i]*np.exp(-g0*(H-Ht[i])/R/(Tstdb[i]+Tb))
    else:
        p = pb[i]*(T/(Tstdb[i]+Tb))**(-g0/A[i]/R)

    # Calculating density
    rho = p/R/T

    # Calculating viscosity with Sutherland's Formula
    mi=mi0*(T0+C)/(T+C)*(T/T0)**(1.5)

    return T,p,rho,mi

#----------------------------------------

def geo_change_sweep(x,y,sweep_x,panel_length,chord_root,chord_tip):

    '''
    This function converts sweep computed at chord fraction x into
    sweep measured at chord fraction y
    (x and y should be between 0 (leading edge) and 1 (trailing edge).
    '''

    sweep_y=np.arctan(np.tan(sweep_x)+(x-y)*(chord_root-chord_tip)/panel_length)

    return sweep_y

#----------------------------------------

def Cf_calc(Mach, altitude, length, rugosity, k_lam, Tba=288.15):
    '''
    This function computes the flat plate friction coefficient
    for a given Reynolds number while taking transition into account

    k_lam: float -> Fraction of the length (from 0 to 1) where
                    transition occurs
    '''
    
    # Dados atmosféricos
    T, p, rho, mi = atmosphere(altitude, Tba)
    

    # Velocidade
    v = np.sqrt(gamma_air*R_air*T)*Mach

    # Reynolds na transição
    Re_conv = rho*v*k_lam*length/mi
    Re_rug = 38.21*(k_lam*length/rugosity)**1.053
    Re_trans = min(Re_conv, Re_rug)

    # Reynolds no fim
    Re_conv = rho*v*length/mi
    if Mach < 0.7:
        Re_rug = 38.21*(length/rugosity)**1.053
    else:
        Re_rug = 44.62*(length/rugosity)**1.053*Mach**1.16
    Re_fim = min(Re_conv, Re_rug)

    # Coeficientes de fricção
    # Laminar na transição
    Cf1 = 1.328/np.sqrt(Re_trans)

    # Turbulento na transição
    Cf2 = 0.455/(np.log10(Re_trans)**2.58*(1+0.144*Mach**2)**0.65)

    # Turbulento no fim
    Cf3 = 0.455/(np.log10(Re_fim)**2.58*(1+0.144*Mach**2)**0.65)

    # Média
    Cf = (Cf1 - Cf2)*k_lam + Cf3

    return Cf

#----------------------------------------

def FF_surface(Mach, tcr, tct, sweep, b, cr, ct, x_c_max_tc=0.4):
    '''
    This function computes the form factor for lifting surfaces

    INPUTS

    tcr: float -> Thickness/chord ratio at the root
    tct: float -> Thickness/chord ratio at the tip
    sweep: float -> Quarter-chord sweep angle [rad]
    b: float -> Wing span (considering both sides. Double this value for vertical tails if necessary)
    cr: float -> Root chord
    ct: float -> Tip chord
    x_c_max_tc: float -> Chord fraction with maximum thickness
    '''

    # Average chord fraction
    t_c = 0.25*tcr + 0.75*tct

    # Sweep at maximum thickness position
    sweep_maxtc=geo_change_sweep(0.25, x_c_max_tc, sweep, b/2, cr, ct)

    # Form factor
    FF = 1.34*Mach**0.18*np.cos(sweep_maxtc)**0.28*(1 + 0.6*t_c/x_c_max_tc + 100*(t_c)**4)

    return FF

#----------------------------------------

def tank_properties(cr_w, ct_w, tcr_w, tct_w, b_w, sweep_w, xr_w,
                    x_tank_c_w, c_tank_c_w, b_tank_b_w_start, b_tank_b_w_end,
                    rho_fuel, gravity):
    '''
    This function computes the maximum fuel tank volume and center of gravity.
    We assume that the tank has a prism shape.

    c_tank_c_w: float -> fraction of the chord where tank begins (0-leading edge, 1-trailing edge)
    c_tank_c_w: float -> fraction of the chord occupied by the tank (between 0 and 1)
    bf_w_start: float -> semi-span fraction where tank begins (0-root, 1-tip)
    bf_w_end: float -> semi-span fraction where tank ends (0-root, 1-tip)
    '''

    # Compute the local chords where the tank begins and ends
    c_tank_start = cr_w + b_tank_b_w_start*(ct_w - cr_w)
    c_tank_end = cr_w + b_tank_b_w_end*(ct_w - cr_w)

    # Compute the local thickness where the tank begins and ends
    tc_tank_start = tcr_w + b_tank_b_w_start*(tct_w - tcr_w)
    tc_tank_end = tcr_w + b_tank_b_w_end*(tct_w - tcr_w)

    # Compute the prism area where the tank begins.
    # We assume that this face is rectangular, and that its height
    # is 85% of the maximum airfoil thickness (Gudmundsson, page 87).
    ll = c_tank_start*c_tank_c_w
    hh = c_tank_start*tc_tank_start*0.85
    S1 = ll*hh

    # Compute the prism area where the tank ends.
    ll = c_tank_end*c_tank_c_w
    hh = c_tank_end*tc_tank_end*0.85
    S2 = ll*hh

    # Compute distance between prism faces along the wing span
    Lprism = 0.5*b_w*(b_tank_b_w_end-b_tank_b_w_start)

    # Compute fuel volume with the prism expression (Torenbeek Fig B-4, pg 448).
    # We multiply by 2 to take into account both semi-wings.
    # The 0.91 factor is to take into account internal structures and fuel expansion,
    # as suggested by Torenbeek.
    V_maxfuel = 0.91*2*Lprism/3*(S1 + S2 + np.sqrt(S1*S2))

    # Compute corresponding fuel weight
    W_maxfuel = V_maxfuel*rho_fuel*gravity

    # Compute the span-wise distance between the first prism face and its center of gravity
    # using the expression from Jenkinson, Fig 7.13, pg 148.
    Lprism_cg = Lprism/4*(S1 + 3*S2 + 2*np.sqrt(S1*S2))/(S1 + S2 + np.sqrt(S1*S2))

    # Now find the span-wise distance between the tank CG and the aircraft centerline
    ycg_fuel = Lprism_cg + 0.5*b_w*b_tank_b_w_start

    # Find the sweep angle at the chord position located on the middle of the chord
    # fraction occupied by the fuel tank
    c_pos = x_tank_c_w + 0.5*c_tank_c_w

    # Sweep at the tank center line
    sweep_tank = geo_change_sweep(0.25, c_pos, sweep_w, b_w/2, cr_w, ct_w)

    # Longitudinal position of the tank CG
    xcg_fuel = xr_w + cr_w*c_pos + ycg_fuel*np.tan(sweep_tank)


    return V_maxfuel, W_maxfuel, xcg_fuel, ycg_fuel

#----------------------------------------

def lin_interp(x0, x1, y0, y1, x):
    '''
    Linear interpolation function
    '''

    y = y0 + (y1-y0)*(x-x0)/(x1-x0)

    return y

#----------------------------------------
#----------------------------------------

#----------------------------------------

def flap_area_fraction(alpha, beta1, beta2, taper):
    '''
    alpha: flap_chord/wing_chord
    beta1: spanwise fraction where flap starts
    beta2: spanwise fraction where flap ends
    taper: taper ratio of the wing
    '''

    S_flap_S_wing = alpha/(1+taper)*(beta2*(2-beta2*(1-taper)) - beta1*(2-beta1*(1-taper)))

    return S_flap_S_wing

#----------------------------------------

def standard_airplane(name='fokker100'):
    '''
    The standard parameters refer to the Fokker 100, but they could be redefined for
    any new aircraft.
    '''

    if name == 'fokker100':

        # This model was taken from measuring the 3D view from
        # https://www.icas.org/ICAS_ARCHIVE/ICAS1988/ICAS-88-1.6.2.pdf
        # Obert, E. The Aerodynamic Development of the Fokker 100


        # Fokker 100
        
        airplane = {'type': 'transport', # Can be 'transport', 'fighter', or 'general'
                    
                    'S_w' : 93.5, # Wing area [m2] - From Obert's paper
                    'AR_w' : 8.32,  # Wing aspect ratio
                    'taper_w' : 0.25, # Wing taper ratio
                    'sweep_w' : 15.76*np.pi/180, # Wing sweep [rad]
                    'dihedral_w' : 3*np.pi/180, # Wing dihedral [rad]
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
                                'Cbase' : 0.839363767995491/3600, # I adjusted this value by hand to match TSFC=0.70 at cruise (This is the value I found for this engine)
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
            
                    'flap_type' : 'double slotted',  # Flap type
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
                    'deltaISA_takeoff' : 15.0, # Variation from ISA standard temperature [ C] - From Obert's paper
                        
                    'altitude_landing' : 0.0, # Altitude for landing computation [m]
                    'distance_landing' : 1340.0, # Required landing distance [m]
                    'deltaISA_landing' : 0.0, # Variation from ISA standard temperature [ C]
                    'MLW_frac' : 40100/43090, # Max Landing Weight / Max Takeoff Weight - From Obert's paper
                        
                    'altitude_cruise' : 35000*ft2m, # Cruise altitude [m] - From Obert's paper
                    'Mach_cruise' : 0.73, # Cruise Mach number - From Obert's paper
                    'range_cruise' : 1310*nm2m, # Cruise range [m] - From Obert's paper

                    'altitude_maxcruise' : 35000*ft2m, # Altitude for high-speed cruise [m]
                    'Mach_maxcruise' : 0.77, # Mach for high-speed cruise [m]
                        
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

    elif name == 'my_airplane_1':

        # This is just a placeholder to register the student airplane.

        airplane = {'type': 'transport', # Can be 'transport', 'fighter', or 'general'
                    
                    'S_w' : 97.58, # Wing area [m2]
                    'AR_w' : 9.57,  # Wing aspect ratio
                    'taper_w' : 0.27, # Wing taper ratio
                    'sweep_w' : 25.1*np.pi/180, # Wing sweep [rad]
                    'dihedral_w' : 4.1*np.pi/180, # Wing dihedral [rad]
                    'xr_w' : 10.5, # Longitudinal position of the wing (with respect to the fuselage nose) [m]
                    'zr_w' : -1.4, # Vertical position of the wing (with respect to the fuselage nose) [m]
                    'tcr_w' : 0.123, # t/c of the root section of the wing
                    'tct_w' : 0.096, # t/c of the tip section of the wing
                    
                    'Cht' : 1.01, # Horizontal tail volume coefficient
                    'Lc_h' : 3.4, # Non-dimensional lever of the horizontal tail (lever/wing_mac)
                    'AR_h' : 4.87, # HT aspect ratio
                    'taper_h' : 0.41, # HT taper ratio
                    'sweep_h' : 28.29*np.pi/180, # HT sweep [rad]
                    'dihedral_h' : 4.76*np.pi/180, # HT dihedral [rad]
                    'zr_h' : 0.75, # Vertical position of the HT [m]
                    'tcr_h' : 0.1, # t/c of the root section of the HT
                    'tct_h' : 0.1, # t/c of the tip section of the HT
                    'eta_h' : 1.0, # Dynamic pressure factor of the HT
                    
                    'Cvt' : 0.069, # Vertical tail volume coefficient
                    'Lb_v' : 0.42, # Non-dimensional lever of the vertical tail (lever/wing_span)
                    'AR_v' : 1.38, # VT aspect ratio
                    'taper_v' : 0.44, # VT taper ratio
                    'sweep_v' : 42.54*np.pi/180, # VT sweep [rad]
                    'zr_v' : 1.6, # Vertical position of the VT [m]
                    'tcr_v' : 0.1, # t/c of the root section of the VT
                    'tct_v' : 0.1, # t/c of the tip section of the VT
                    
                    'L_f' : 30.85, # Fuselage length [m]
                    'D_f' : 3.7, # Fuselage diameter [m]
                    
                    'x_n' : 10, # Longitudinal position of the nacelle frontal face [m]
                    'y_n' : 5.01, # Lateral position of the nacelle centerline [m]
                    'z_n' : -2.35, # Vertical position of the nacelle centerline [m]
                    'L_n' : 4.91, # Nacelle length [m]
                    'D_n' : 1.69, # Nacelle diameter [m]
                    
                    'n_engines' : 2, # Number of engines
                    'n_engines_under_wing' : 0, # Number of engines installed under the wing
                    'engine' : {'model' : 'Howe turbofan', # Check engineTSFC function for options
                                'BPR' : 13, # Engine bypass ratio
                                'Cbase' : 0.7/3600, # I adjusted this value by hand to match the fuel weight
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
        
                    'flap_type' : 'double slotted',  # Flap type
                    'c_flap_c_wing' : 0.30, # Fraction of the wing chord occupied by flaps
                    'b_flap_b_wing' : 0.60, # Fraction of the wing span occupied by flaps (including fuselage portion)
                    
                    'slat_type' : None, # Slat type
                    'c_slat_c_wing' : 0.00, # Fraction of the wing chord occupied by slats
                    'b_slat_b_wing' : 0.00, # Fraction of the wing span occupied by slats

                    'c_ail_c_wing' : 0.27, # Fraction of the wing chord occupied by aileron
                    'b_ail_b_wing' : 0.34, # Fraction of the wing span occupied by aileron
                    
                    'h_ground' : 35.0*ft2m, # Distance to the ground for ground effect computation [m]
                    'k_exc_drag' : 0.03, # Excrescence drag factor

                    'winglet' : False, # Add winglet
                    
                    'altitude_takeoff' : 0.0, # Altitude for takeoff computation [m]
                    'distance_takeoff' : 1800.0, # Required takeoff distance [m]
                    
                    'altitude_landing' : 0.0, # Altitude for landing computation [m]
                    'distance_landing' : 1800.0, # Required landing distance [m] (The actual Fokker100 distance is 1350 m but it is very restrictive compared to the historical regression. Therefore I kept the same TO distance since the aircraft should takeoff and land at the same runway)
                    
                    'altitude_cruise' : 35000*ft2m, # Cruise altitude [m]
                    'Mach_cruise' : 0.8, # Cruise Mach number
                    'range_cruise' : 2200*nm2m, # Cruise range [m]
                    
                    'loiter_time' : 45*60, # Loiter time [s]
                    
                    'altitude_altcruise' : 4572, # Alternative cruise altitude [m]
                    'Mach_altcruise' : 0.5, # Alternative cruise Mach number
                    'range_altcruise' : 200*nm2m, # Alternative cruise range [m]
                    
                    'W_payload' : 10000*gravity, # Payload weight [N]
                    'xcg_payload' : 14.4, # Longitudinal position of the Payload center of gravity [m]
                    
                    'W_crew' : 4*91*gravity, # Crew weight [N]
                    'xcg_crew' : 2.5, # Longitudinal position of the Crew center of gravity [m]

                    'block_range' : 400*nm2m, # Block range [m]
                    'block_time' : (1.0 + 2*40/60)*3600, # Block time [s]
                    'n_captains' : 1, # Number of captains in flight
                    'n_copilots' : 1, # Number of copilots in flight
                    
                    'rho_fuel' : 804, # Fuel density kg/m3 (This is Jet A-1)

                    'W0_guess' : 50150*gravity, # Guess for MTOW
                    'T0_guess' : 0.3*50150*gravity, # 30% OF W0_GUES!!!!!!!!
                    'MLW_frac' : 0.85 # Max Landing Weight / Max Takeoff Weight
                    }
        
    elif name == 'my_airplane_2':

        # This is just a placeholder to register the student airplane.

        airplane = {'type': 'transport', # Can be 'transport', 'fighter', or 'general'
                    
                    'S_w' : 105.58, # Wing area [m2]
                    'AR_w' : 10.32,  # Wing aspect ratio
                    'taper_w' : 0.25, # Wing taper ratio
                    'sweep_w' : 26.4*np.pi/180, # Wing sweep [rad]
                    'dihedral_w' : 4.5*np.pi/180, # Wing dihedral [rad]
                    'xr_w' : 14.35, # Longitudinal position of the wing (with respect to the fuselage nose) [m]
                    'zr_w' : -1.3, # Vertical position of the wing (with respect to the fuselage nose) [m]
                    'tcr_w' : 0.123, # t/c of the root section of the wing
                    'tct_w' : 0.096, # t/c of the tip section of the wing
                    
                    'Cht' : 0.92, # Horizontal tail volume coefficient
                    'Lc_h' : 5, # Non-dimensional lever of the horizontal tail (lever/wing_mac)
                    'AR_h' : 4.87, # HT aspect ratio
                    'taper_h' : 0.41, # HT taper ratio
                    'sweep_h' : 28.29*np.pi/180, # HT sweep [rad]
                    'dihedral_h' : 4.76*np.pi/180, # HT dihedral [rad]
                    'zr_h' : 6.35, # Vertical position of the HT [m]
                    'tcr_h' : 0.1, # t/c of the root section of the HT
                    'tct_h' : 0.1, # t/c of the tip section of the HT
                    'eta_h' : 1.0, # Dynamic pressure factor of the HT
                    
                    'Cvt' : 0.084, # Vertical tail volume coefficient 0.069
                    'Lb_v' : 0.42, # Non-dimensional lever of the vertical tail (lever/wing_span)
                    'AR_v' : 1.38, # VT aspect ratio
                    'taper_v' : 0.44, # VT taper ratio
                    'sweep_v' : 42.54*np.pi/180, # VT sweep [rad]
                    'zr_v' : 1.26, # Vertical position of the VT [m]
                    'tcr_v' : 0.1, # t/c of the root section of the VT
                    'tct_v' : 0.1, # t/c of the tip section of the VT
                    
                    'L_f' : 36.38, # Fuselage length [m]
                    'D_f' : 3.4, # Fuselage diameter [m]
                    
                    'x_n' : 23.68, # Longitudinal position of the nacelle frontal face [m]
                    'y_n' : 2.51, # Lateral position of the nacelle centerline [m]
                    'z_n' : 0.45, # Vertical position of the nacelle centerline [m]
                    'L_n' : 4.91, # Nacelle length [m]
                    'D_n' : 1.69, # Nacelle diameter [m]
                    
                    'n_engines' : 2, # Number of engines
                    'n_engines_under_wing' : 0, # Number of engines installed under the wing
                    'engine' : {'model' : 'Howe turbofan', # Check engineTSFC function for options
                                'BPR' : 6, # Engine bypass ratio
                                'Cbase' : 0.7/3600, # I adjusted this value by hand to match the fuel weight
                                },
                    
                    'x_nlg' : 3.7, # Longitudinal position of the nose landing gear [m]
                    'x_mlg' : 19.5, # Longitudinal position of the main landing gear [m]
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
        
                    'flap_type' : 'double slotted',  # Flap type
                    'c_flap_c_wing' : 0.30, # Fraction of the wing chord occupied by flaps
                    'b_flap_b_wing' : 0.60, # Fraction of the wing span occupied by flaps (including fuselage portion)
                    
                    'slat_type' : None, # Slat type
                    'c_slat_c_wing' : 0.00, # Fraction of the wing chord occupied by slats
                    'b_slat_b_wing' : 0.00, # Fraction of the wing span occupied by slats

                    'c_ail_c_wing' : 0.27, # Fraction of the wing chord occupied by aileron
                    'b_ail_b_wing' : 0.34, # Fraction of the wing span occupied by aileron
                    
                    'h_ground' : 35.0*ft2m, # Distance to the ground for ground effect computation [m]
                    'k_exc_drag' : 0.03, # Excrescence drag factor

                    'winglet' : False, # Add winglet
                    
                    'altitude_takeoff' : 0.0, # Altitude for takeoff computation [m]
                    'distance_takeoff' : 1800.0, # Required takeoff distance [m]
                    
                    'altitude_landing' : 0.0, # Altitude for landing computation [m]
                    'distance_landing' : 1800.0, # Required landing distance [m] (The actual Fokker100 distance is 1350 m but it is very restrictive compared to the historical regression. Therefore I kept the same TO distance since the aircraft should takeoff and land at the same runway)
                    
                    'altitude_cruise' : 35000*ft2m, # Cruise altitude [m]
                    'Mach_cruise' : 0.8, # Cruise Mach number
                    'range_cruise' : 2200*nm2m, # Cruise range [m]
                    
                    'loiter_time' : 45*60, # Loiter time [s]
                    
                    'altitude_altcruise' : 4572, # Alternative cruise altitude [m]
                    'Mach_altcruise' : 0.5, # Alternative cruise Mach number
                    'range_altcruise' : 200*nm2m, # Alternative cruise range [m]
                    
                    'W_payload' : 10000*gravity, # Payload weight [N]
                    'xcg_payload' : 14.4, # Longitudinal position of the Payload center of gravity [m]
                    
                    'W_crew' : 4*91*gravity, # Crew weight [N]
                    'xcg_crew' : 2.5, # Longitudinal position of the Crew center of gravity [m]

                    'block_range' : 400*nm2m, # Block range [m]
                    'block_time' : (1.0 + 2*40/60)*3600, # Block time [s]
                    'n_captains' : 1, # Number of captains in flight
                    'n_copilots' : 1, # Number of copilots in flight
                    
                    'rho_fuel' : 804, # Fuel density kg/m3 (This is Jet A-1)

                    'W0_guess' : 49200*gravity, # Guess for MTOW
                    'T0_guess' : 0.3*49200*gravity, # 30% OF W0_GUES!!!!!!!!
                    'MLW_frac' : 0.85 # Max Landing Weight / Max Takeoff Weight
                    }
    

    return airplane

#----------------------------------------

def plot3d(airplane, figname='3dview.png', az1=45, az2=-135):
    '''
    az1 and az2: degrees of azimuth and elevation for the 3d plot view
    '''

    from matplotlib.patches import Ellipse
    import mpl_toolkits.mplot3d.art3d as art3d

    xr_w = airplane['xr_w']
    zr_w = airplane['zr_w']
    b_w = airplane['b_w']

    tct_w = airplane['tct_w']
    tcr_w = airplane['tcr_w']

    cr_w = airplane['cr_w']
    xt_w = airplane['xt_w']
    yt_w = airplane['yt_w']
    zt_w = airplane['zt_w']
    ct_w = airplane['ct_w']

    xr_h = airplane['xr_h']
    zr_h = airplane['zr_h']

    tcr_h = airplane['tcr_h']
    tct_h = airplane['tct_h']

    cr_h = airplane['cr_h']
    xt_h = airplane['xt_h']
    yt_h = airplane['yt_h']
    zt_h = airplane['zt_h']
    ct_h = airplane['ct_h']
    b_h  = airplane['b_h']

    xr_v = airplane['xr_v']
    zr_v = airplane['zr_v']

    tcr_v = airplane['tcr_v']
    tct_v = airplane['tct_v']

    cr_v = airplane['cr_v']
    xt_v = airplane['xt_v']
    zt_v = airplane['zt_v']
    ct_v = airplane['ct_v']
    b_v  = airplane['b_v']

    L_f = airplane['L_f']
    D_f = airplane['D_f']
    x_n = airplane['x_n']
    y_n = airplane['y_n']
    z_n = airplane['z_n']
    L_n = airplane['L_n']
    D_n = airplane['D_n']

    has_winglet = airplane['winglet']

    if 'xcg_fwd' in airplane:
        xcg_fwd = airplane['xcg_fwd']
        xcg_aft = airplane['xcg_aft']
    else:
        xcg_fwd = None
        xcg_aft = None

    if 'xnp' in airplane:
        xnp = airplane['xnp']
    else:
        xnp = None

    x_nlg = airplane['x_nlg']
    y_nlg = 0
    z_nlg = airplane['z_lg']
    x_mlg = airplane['x_mlg']
    y_mlg = airplane['y_mlg']
    z_mlg = airplane['z_lg']
    x_tailstrike = airplane['x_tailstrike']
    z_tailstrike = airplane['z_tailstrike']

    flap_type = airplane['flap_type']
    b_flap_b_wing = airplane['b_flap_b_wing']
    c_flap_c_wing = airplane['c_flap_c_wing']

    slat_type = airplane['slat_type']
    b_slat_b_wing = airplane['b_slat_b_wing']
    c_slat_c_wing = airplane['c_slat_c_wing']

    b_ail_b_wing = airplane['b_ail_b_wing']
    c_ail_c_wing = airplane['c_ail_c_wing']

    ### PLOT

    #fig = plt.figure(fignum,figsize=(20, 10))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.set_aspect('equal')

    ax.plot([xr_w, xt_w, xt_w+ct_w, xr_w+cr_w, xt_w+ct_w, xt_w, xr_w],
            [0.0, yt_w, yt_w, 0.0, -yt_w, -yt_w, 0.0],
            [zr_w+cr_w*tcr_w/2, zt_w+ct_w*tct_w/2, zt_w+ct_w*tct_w/2, zr_w+cr_w*tcr_w/2, zt_w+ct_w*tct_w/2, zt_w+ct_w*tct_w/2, zr_w+cr_w*tcr_w/2],color='blue')

    if has_winglet:
        ttw = 0.21 # Winglet taper ratio
        ax.plot([xt_w, xt_w + (1-ttw)*ct_w, xt_w+ct_w, xt_w+ct_w, xt_w],
            [yt_w, yt_w, yt_w, yt_w, yt_w],
            [zt_w, zt_w+ct_w, zt_w+ct_w, zt_w, zt_w],color='blue')
        ax.plot([xt_w, xt_w + (1-ttw)*ct_w, xt_w+ct_w, xt_w+ct_w, xt_w],
            [-yt_w, -yt_w, -yt_w, -yt_w, -yt_w],
            [zt_w, zt_w+ct_w, zt_w+ct_w, zt_w, zt_w],color='blue')

    ax.plot([xr_h, xt_h, xt_h+ct_h, xr_h+cr_h, xt_h+ct_h, xt_h, xr_h],
            [0.0, yt_h, yt_h, 0.0, -yt_h, -yt_h, 0.0],
            [zr_h+cr_h*tcr_h/2, zt_h+ct_h*tct_h/2, zt_h+ct_h*tct_h/2, zr_h+cr_h*tcr_h/2, zt_h+ct_h*tct_h/2, zt_h+ct_h*tct_h/2, zr_h+cr_h*tcr_h/2],color='green')


    ax.plot([xr_v        , xt_v        , xt_v+ct_v   , xr_v+cr_v   , xr_v        ],
            [tcr_v*cr_v/2, tct_v*ct_v/2, tct_v*ct_v/2, tcr_v*cr_v/2, tcr_v*cr_v/2],
            [zr_v        , zt_v        , zt_v        , zr_v        , zr_v        ],\
            color='orange')

    ax.plot([ xr_v        ,  xt_v        ,  xt_v+ct_v   ,  xr_v+cr_v   ,  xr_v        ],
            [-tcr_v*cr_v/2, -tct_v*ct_v/2, -tct_v*ct_v/2, -tcr_v*cr_v/2, -tcr_v*cr_v/2],
            [ zr_v        ,  zt_v        ,  zt_v        ,  zr_v        ,  zr_v     ],\
            color='orange')



    ax.plot([0.0, L_f],
            [0.0, 0.0],
            [0.0, 0.0])
    ax.plot([x_n, x_n+L_n],
            [y_n, y_n],
            [z_n, z_n])
    ax.plot([x_n, x_n+L_n],
            [-y_n, -y_n],
            [z_n, z_n])

    # Forward CG point
    if xcg_fwd is not None:
        ax.plot([xcg_fwd], [0.0], [0.0],'ko')
    
    # Rear CG point
    if xcg_aft is not None:
        ax.plot([xcg_aft], [0.0], [0.0],'ko')
    
    # Neutral point
    if xnp is not None:
        ax.plot([xnp], [0.0], [0.0],'x')

    # Define a parametrized fuselage by setting height and width
    # values along its axis
    # xx is non-dimensionalized by fuselage length
    # hh and ww are non-dimensionalized by fuselage diameter
    # There are 6 stations where we define the arrays:
    # nose1; nose2; nose3; cabin start; tailstrike; tail
    xx = [0.0, 1.24/41.72, 3.54/41.72, 7.55/41.72, x_tailstrike/L_f, 1.0]
    hh = [0.0, 2.27/4.0, 3.56/4.0, 1.0, 1.0, 1.07/4.0]
    ww = [0.0, 1.83/4.0, 3.49/4.0, 1.0, 1.0, 0.284/4]
    num_tot_ell = 50 # Total number of ellipses
    
    # Loop over every section
    for ii in range(len(xx)-1):
        
        # Define number of ellipses based on the section length
        num_ell = int((xx[ii+1]-xx[ii])*num_tot_ell)+1
        
        # Define arrays of dimensional positions, heights and widths
        # for the current section
        xdim = np.linspace(xx[ii], xx[ii+1], num_ell)*L_f
        hdim = np.linspace(hh[ii], hh[ii+1], num_ell)*D_f
        wdim = np.linspace(ww[ii], ww[ii+1], num_ell)*D_f
        
        # Loop over every ellipse
        for xc, hc, wc in zip(xdim, hdim, wdim):

            # Define ellipse center to make flat top at the fuselage tail
            if xc > x_tailstrike:
                yye = (D_f-hc)/2
            else:
                yye = 0

            p = Ellipse((0, yye), wc, hc, angle=0,
                        facecolor = 'none', edgecolor = 'k', lw=1.0)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=xc, zdir="x")


    #____________________________________________________________
    #                                                            \
    # MLG / NLG
    
    # Check if LG is activated
    d_lg = 0
    if x_nlg is not None:
    
        # Make landing gear dimensions based on the fuselage
        w_lg = 0.05*D_f
        d_lg = 4*w_lg
        
        mlg_len = np.linspace(y_mlg-w_lg/2, y_mlg+w_lg/2, 2)
        nlg_len = np.linspace(y_nlg-w_lg/2, y_nlg+w_lg/2, 2)
        
        for i in range(len(mlg_len)):
            p = Ellipse((x_mlg, z_mlg), d_lg, d_lg, angle=0,\
            facecolor = 'gray', edgecolor = 'k', lw=2)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=mlg_len[i], zdir="y")
            
            p = Ellipse((x_mlg, z_mlg), d_lg, d_lg, angle=0,\
            facecolor = 'gray', edgecolor = 'k', lw=2)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-mlg_len[i], zdir="y")
    
            # NLG
            p = Ellipse((x_nlg, z_nlg), d_lg, d_lg, angle=0,\
            facecolor = 'gray', edgecolor = 'k', lw=1.5)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=nlg_len[i], zdir="y")

    # Nacelle
    nc_len = np.linspace(x_n,x_n+L_n,11)
    for i in range(len(nc_len)):
        p = Ellipse((y_n, z_n), D_n, D_n, angle=0,\
        facecolor = 'none', edgecolor = 'orange', lw=1.0)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=nc_len[i], zdir="x")

        # Inner wall
        #p = Ellipse((y_n, z_n), D_n*0.8, D_n*0.8, angle=0,\
        #facecolor = 'none', edgecolor = 'k', lw=.1)
        #ax.add_patch(p)
        #art3d.pathpatch_2d_to_3d(p, z=nc_len[i], zdir="x")


        p = Ellipse((-y_n, z_n), D_n, D_n, angle=0, \
        facecolor = 'none', edgecolor = 'orange', lw=1.0)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=nc_len[i], zdir="x")

        # Inner wall
        #p = Ellipse((-y_n, z_n), D_n*0.8, D_n*0.8, angle=0, \
        #facecolor = 'none', edgecolor = 'k', lw=.1)
        #ax.add_patch(p)
        #art3d.pathpatch_2d_to_3d(p, z=nc_len[i], zdir="x")

    # Aileron
    ail_tip_margin = 0.02 # Margem entre flap e aileron em % de b_w

    # Spanwise positions (root and tip)
    yr_a = (1.0 - (ail_tip_margin + b_ail_b_wing))*b_w/2
    yt_a = (1.0 - (ail_tip_margin))*b_w/2

    cr_a = lin_interp(0, b_w/2, cr_w, ct_w, yr_a)*c_ail_c_wing
    ct_a = lin_interp(0, b_w/2, cr_w, ct_w, yt_a)*c_ail_c_wing

    # To find the longitudinal position of the aileron LE, we find the TE position first
    # then we subtract the chord
    xr_a = lin_interp(0, b_w/2, xr_w+cr_w, xt_w+ct_w, yr_a) - cr_a
    xt_a = lin_interp(0, b_w/2, xr_w+cr_w, xt_w+ct_w, yt_a) - ct_a

    zr_a = lin_interp(0, b_w/2, zr_w, zt_w, yr_a)
    zt_a = lin_interp(0, b_w/2, zr_w, zt_w, yt_a)

    # Airfoil thickness at aileron location
    tcr_a = lin_interp(0, b_w/2, tcr_w, tct_w, yr_a)
    tct_a = lin_interp(0, b_w/2, tcr_w, tct_w, yt_a)

    ax.plot([xr_a, xt_a, xt_a+ct_a, xr_a+cr_a, xr_a],
            [yr_a, yt_a, yt_a     , yr_a     , yr_a],
            [zr_a+cr_a*tcr_a/2/c_ail_c_wing, zt_a+ct_a*tct_a/2/c_ail_c_wing, zt_a+ct_a*tct_a/2/c_ail_c_wing     , zr_a+cr_a*tcr_a/2/c_ail_c_wing, zr_a+cr_a*tcr_a/2/c_ail_c_wing],lw=1,color='green')

    ax.plot([ xr_a,  xt_a,  xt_a+ct_a,  xr_a+cr_a,  xr_a],
            [-yr_a, -yt_a, -yt_a     , -yr_a     , -yr_a],
            [ zr_a+cr_a*tcr_a/2/c_ail_c_wing,  zt_a+ct_a*tct_a/2/c_ail_c_wing,  zt_a+ct_a*tct_a/2/c_ail_c_wing,  zr_a+cr_a*tcr_a/2/c_ail_c_wing     ,  zr_a+cr_a*tcr_a/2/c_ail_c_wing],lw=1,color='green')

    # Fuel tank
    c_tank_c_w = airplane['c_tank_c_w']
    x_tank_c_w = airplane['x_tank_c_w']
    b_tank_b_w_start = airplane['b_tank_b_w_start']
    b_tank_b_w_end = airplane['b_tank_b_w_end']

    # Spanwise positions (root and tip)
    yr_tk = b_tank_b_w_start*b_w/2
    yt_tk = b_tank_b_w_end*b_w/2

    cr_tk = lin_interp(0, b_w/2, cr_w, ct_w, yr_tk)*c_tank_c_w
    ct_tk = lin_interp(0, b_w/2, cr_w, ct_w, yt_tk)*c_tank_c_w

    # To find the longitudinal position of the tank LE
    xr_tk = lin_interp(0, b_w/2, xr_w, xt_w, yr_tk) + cr_tk*x_tank_c_w/c_tank_c_w
    xt_tk = lin_interp(0, b_w/2, xr_w, xt_w, yt_tk) + ct_tk*x_tank_c_w/c_tank_c_w

    zr_tk = lin_interp(0, b_w/2, zr_w, zt_w, yr_tk)
    zt_tk = lin_interp(0, b_w/2, zr_w, zt_w, yt_tk)

    # Airfoil thickness at tank location
    tcr_tk = lin_interp(0, b_w/2, tcr_w, tct_w, yr_tk)
    tct_tk = lin_interp(0, b_w/2, tcr_w, tct_w, yt_tk)

    ax.plot([xr_tk, xt_tk, xt_tk+ct_tk, xr_tk+cr_tk, xr_tk],
            [yr_tk, yt_tk, yt_tk     , yr_tk     , yr_tk],
            [zr_tk+cr_tk*tcr_tk/2, zt_tk+ct_tk*tct_tk/2, zt_tk+ct_tk*tct_tk/2     , zr_tk+cr_tk*tcr_tk/2, zr_tk+cr_tk*tcr_tk/2],lw=1,color='magenta')

    ax.plot([ xr_tk,  xt_tk,  xt_tk+ct_tk,  xr_tk+cr_tk,  xr_tk],
            [-yr_tk, -yt_tk, -yt_tk     , -yr_tk     , -yr_tk],
            [ zr_tk+cr_tk*tcr_tk/2,  zt_tk+ct_tk*tct_tk/2,  zt_tk+ct_tk*tct_tk/2,  zr_tk+cr_tk*tcr_tk/2     ,  zr_tk+cr_tk*tcr_tk/2],lw=1,color='magenta')

    # Slat
    if slat_type is not None:
        
        #slat_tip_margin = 0.02  # Margem da ponta como % da b_w
        #slat_root_margin = 0.12 # Margem da raiz como % da b_w
        #hist_c_s = 0.25        # Corda do Flap
        #hist_b_s = 1 - slat_root_margin - slat_tip_margin

        # Spanwise positions (root and tip)
        yr_s = D_f/2
        yt_s = b_slat_b_wing*b_w/2

        cr_s = lin_interp(0, b_w/2, cr_w, ct_w, yr_s)*c_slat_c_wing
        ct_s = lin_interp(0, b_w/2, cr_w, ct_w, yt_s)*c_slat_c_wing

        # Find the longitudinal position of the slat LE
        xr_s = lin_interp(0, b_w/2, xr_w, xt_w, yr_s)
        xt_s = lin_interp(0, b_w/2, xr_w, xt_w, yt_s)

        zr_s = lin_interp(0, b_w/2, zr_w, zt_w, yr_s)
        zt_s = lin_interp(0, b_w/2, zr_w, zt_w, yt_s)

        # Airfoil thickness at slat location
        tcr_s = lin_interp(0, b_w/2, tcr_w, tct_w, yr_s)
        tct_s = lin_interp(0, b_w/2, tcr_w, tct_w, yt_s)


        ax.plot([xr_s, xt_s, xt_s+ct_s, xr_s+cr_s, xr_s],
                [yr_s, yt_s, yt_s     , yr_s     , yr_s],
                [zr_s+cr_s*tcr_s/2/c_slat_c_wing, zt_s+ct_s*tct_s/2/c_slat_c_wing, zt_s+ct_s*tct_s/2/c_slat_c_wing     , zr_s+cr_s*tcr_s/2/c_slat_c_wing, zr_s+cr_s*tcr_s/2/c_slat_c_wing],lw=1,color='m')

        ax.plot([ xr_s,  xt_s,  xt_s+ct_s,  xr_s+cr_s,  xr_s],
                [-yr_s, -yt_s, -yt_s     , -yr_s     , -yr_s],
                [ zr_s+cr_s*tcr_s/2/c_slat_c_wing,  zt_s+ct_s*tct_s/2/c_slat_c_wing,  zt_s+ct_s*tct_s/2/c_slat_c_wing,  zr_s+cr_s*tcr_s/2/c_slat_c_wing     ,  zr_s+cr_s*tcr_s/2/c_slat_c_wing],lw=1,color='m')

    # Flap outboard
    if flap_type is not None:

        # Spanwise positions (root and tip)
        yr_f = D_f/2
        yt_f = b_flap_b_wing*b_w/2

        cr_f = lin_interp(0, b_w/2, cr_w, ct_w, yr_f)*c_flap_c_wing
        ct_f = lin_interp(0, b_w/2, cr_w, ct_w, yt_f)*c_flap_c_wing

        # To find the longitudinal position of the flap LE, we find the TE position first
        # then we subtract the chord
        xr_f = lin_interp(0, b_w/2, xr_w+cr_w, xt_w+ct_w, yr_f) - cr_f
        xt_f = lin_interp(0, b_w/2, xr_w+cr_w, xt_w+ct_w, yt_f) - ct_f

        zr_f = lin_interp(0, b_w/2, zr_w, zt_w, yr_f)
        zt_f = lin_interp(0, b_w/2, zr_w, zt_w, yt_f)

        # Airfoil thickness at flap location
        tcr_f = lin_interp(0, b_w/2, tcr_w, tct_w, yr_f)
        tct_f = lin_interp(0, b_w/2, tcr_w, tct_w, yt_f)


        ax.plot([xr_f, xt_f, xt_f+ct_f, xr_f+cr_f, xr_f],
                [yr_f, yt_f, yt_f     , yr_f     , yr_f],
                [zr_f+cr_f*tcr_f/2/c_flap_c_wing, zt_f+ct_f*tct_f/2/c_flap_c_wing, zt_f+ct_f*tct_f/2/c_flap_c_wing     , zr_f+cr_f*tcr_f/2/c_flap_c_wing, zr_f+cr_f*tcr_f/2/c_flap_c_wing],lw=1,color='r')

        ax.plot([ xr_f,  xt_f,  xt_f+ct_f,  xr_f+cr_f,  xr_f],
                [-yr_f, -yt_f, -yt_f     , -yr_f     , -yr_f],
                [ zr_f+cr_f*tcr_f/2/c_flap_c_wing,  zt_f+ct_f*tct_f/2/c_flap_c_wing,  zt_f+ct_f*tct_f/2/c_flap_c_wing,  zr_f+cr_f*tcr_f/2/c_flap_c_wing     ,  zr_f+cr_f*tcr_f/2/c_flap_c_wing],lw=1,color='r')

    # Elevator
    ele_tip_margin = 0.1  # Margem do profundor para a ponta
    ele_root_margin = 0.1 # Margem do profundor para a raiz
    hist_b_e = 1-ele_root_margin-ele_tip_margin
    hist_c_e = 0.25


    ct_e_loc = (1-ele_tip_margin)*(ct_h - cr_h)+cr_h
    cr_e_loc = (1-hist_b_e-ele_tip_margin)*(ct_h - cr_h)+cr_h

    ct_e = ct_e_loc*hist_c_e
    cr_e = cr_e_loc*hist_c_e

    xr_e = (1-hist_b_e-ele_tip_margin)*(xt_h - xr_h)+xr_h + cr_e_loc*(1-hist_c_e)
    xt_e = (1-ele_tip_margin)*(xt_h - xr_h)+xr_h + ct_e_loc*(1-hist_c_e)

    yr_e = (1-hist_b_e-ele_tip_margin)*b_h/2
    yt_e = (1-ele_tip_margin)*b_h/2

    zr_e = (1-hist_b_e-ele_tip_margin)*(zt_h - zr_h)+zr_h
    zt_e = (1-ele_tip_margin)*(zt_h - zr_h)+zr_h



    ax.plot([xr_e, xt_e, xt_e+ct_e, xr_e+cr_e, xr_e],
            [yr_e, yt_e, yt_e     , yr_e     , yr_e],
            [zr_e, zt_e, zt_e     , zr_e     , zr_e],lw=1,color='g')

    ax.plot([ xr_e,  xt_e,  xt_e+ct_e,  xr_e+cr_e,  xr_e],
            [-yr_e, -yt_e, -yt_e     , -yr_e     , -yr_e],
            [ zr_e,  zt_e,  zt_e     ,  zr_e     ,  zr_e],lw=1,color='g')

    # Rudder
    ver_base_margin = 0.1               # Local da base % de b_v
    ver_tip_margin1 = 0.1               # Local da base % de b_v
    ver_tip_margin = 1-ver_tip_margin1  # Local do topo % de b_v
    hist_c_v = 0.32

    cr_v_loc = ver_base_margin*(ct_v - cr_v)+cr_v
    ct_v_loc = ver_tip_margin*(ct_v - cr_v)+cr_v


    cr_v2 = cr_v_loc*hist_c_v
    ct_v2 = ct_v_loc*hist_c_v


    xr_v2 = ver_base_margin*(xt_v - xr_v)+xr_v+cr_v_loc*(1-hist_c_v)
    xt_v2 = ver_tip_margin*(xt_v - xr_v)+xr_v+ct_v_loc*(1-hist_c_v)


    zr_v2 = ver_base_margin*(zt_v - zr_v)+zr_v
    zt_v2 = ver_tip_margin*(zt_v - zr_v)+zr_v



    ax.plot([xr_v2  , xt_v2  , xt_v2+ct_v2   , xr_v2+cr_v2   , xr_v2        ],
            [tcr_v*cr_v_loc/2, tct_v*ct_v_loc/2, tct_v*ct_v_loc/2, \
            tcr_v*cr_v_loc/2, tcr_v*cr_v_loc/2],
            [zr_v2  , zt_v2   , zt_v2       , zr_v2        , zr_v2        ],\
            color='orange')


    ax.plot([xr_v2  , xt_v2  , xt_v2+ct_v2   , xr_v2+cr_v2   , xr_v2        ],
            [-tcr_v*cr_v_loc/2, -tct_v*ct_v_loc/2, -tct_v*ct_v_loc/2, \
            -tcr_v*cr_v_loc/2, -tcr_v*cr_v_loc/2],
            [zr_v2  , zt_v2   , zt_v2       , zr_v2        , zr_v2        ],\
            color='orange')

    # _______ONLY FRONT VIEW_______

    # Wing Lower
    #------------------------------
    ax.plot([xr_w    , xt_w, xt_w+ct_w, xr_w+cr_w, xt_w+ct_w, xt_w, xr_w],
            [0.0     , yt_w, yt_w, 0.0, -yt_w, -yt_w, 0.0],
            [zr_w-tcr_w*cr_w/2, zt_w-tct_w*ct_w/2, zt_w-tct_w*ct_w/2, zr_w-tcr_w*cr_w/2, \
             zt_w-tct_w*ct_w/2, zt_w-tct_w*ct_w/2, zr_w-tcr_w*cr_w/2],color='blue')

    ax.plot([xr_w         , xr_w],
            [0.0          , 0.0 ],
            [zr_w-tcr_w*cr_w/2, zr_w+tcr_w*cr_w/2],color='blue')
    ax.plot([xr_w+cr_w         , xr_w+cr_w],
            [0.0          , 0.0 ],
            [zr_w-tcr_w*cr_w/2, zr_w+tcr_w*cr_w/2],color='blue')

    ax.plot([xt_w         , xt_w],
            [yt_w         , yt_w ],
            [zt_w-tct_w*ct_w/2, zt_w+tct_w*ct_w/2],color='blue')
    ax.plot([xt_w+ct_w    , xt_w+ct_w],
            [yt_w         , yt_w ],
            [zt_w-tct_w*ct_w/2, zt_w+tct_w*ct_w/2],color='blue')

    ax.plot([xt_w         , xt_w],
            [-yt_w         , -yt_w ],
            [zt_w-tct_w*ct_w/2, zt_w+tct_w*ct_w/2],color='blue')
    ax.plot([xt_w+ct_w    , xt_w+ct_w],
            [-yt_w         , -yt_w ],
            [zt_w-tct_w*ct_w/2, zt_w+tct_w*ct_w/2],color='blue')

    #------------------------------



    # HT Lower
    #------------------------------
    ax.plot([xr_h    , xt_h, xt_h+ct_h, xr_h+cr_h, xt_h+ct_h, xt_h, xr_h],
            [0.0     , yt_h, yt_h, 0.0, -yt_h, -yt_h, 0.0],
            [zr_h-tcr_h*cr_h/2, zt_h-tct_h*ct_h/2, zt_h-tct_h*ct_h/2, zr_h-tcr_h*cr_h/2, \
             zt_h-tct_h*ct_h/2, zt_h-tct_h*ct_h/2, zr_h-tcr_h*cr_h/2],color='green')

    ax.plot([xr_h         , xr_h],
            [0.0          , 0.0 ],
            [zr_h-tcr_h*cr_h/2, zr_h+tcr_h*cr_h/2],color='green')
    ax.plot([xr_h+cr_h         , xr_h+cr_h],
            [0.0          , 0.0 ],
            [zr_h-tcr_h*cr_h/2, zr_h+tcr_h*cr_h/2],color='green')

    ax.plot([xt_h         , xt_h],
            [yt_h         , yt_h ],
            [zt_h-tct_h*ct_h/2, zt_h+tct_h*ct_h/2],color='green')
    ax.plot([xt_h+ct_h    , xt_h+ct_h],
            [yt_h         , yt_h ],
            [zt_h-tct_h*ct_h/2, zt_h+tct_h*ct_h/2],color='green')

    ax.plot([ xt_h         ,  xt_h],
            [-yt_h         , -yt_h ],
            [ zt_h-tct_h*ct_h/2, zt_h+tct_h*ct_h/2],color='green')
    ax.plot([ xt_h+ct_h    ,  xt_h+ct_h],
            [-yt_h         , -yt_h ],
            [ zt_h-tct_h*ct_h/2, zt_h+tct_h*ct_h/2],color='green')


    # Slat Lower
    #------------------------------
    if slat_type is not None:
        ax.plot([xr_s, xt_s, xt_s+ct_s, xr_s+cr_s, xr_s],
                [yr_s, yt_s, yt_s     , yr_s     , yr_s],
                [zr_s-tcr_s*cr_s/2/c_slat_c_wing ,\
                 zt_s-tct_s*ct_s/2/c_slat_c_wing ,\
                 zt_s-tct_s*ct_s/2/c_slat_c_wing ,\
                 zr_s-tcr_s*cr_s/2/c_slat_c_wing ,\
                 zr_s-tcr_s*cr_s/2/c_slat_c_wing],\
                 lw=1,color='m')

        ax.plot([ xr_s,  xt_s,  xt_s+ct_s,  xr_s+cr_s,  xr_s],
                [-yr_s, -yt_s, -yt_s     , -yr_s     , -yr_s],
                [ zr_s-tcr_s*cr_s/2/c_slat_c_wing,\
                  zt_s-tct_s*ct_s/2/c_slat_c_wing,\
                  zt_s-tct_s*ct_s/2/c_slat_c_wing,\
                  zr_s-tcr_s*cr_s/2/c_slat_c_wing,\
                  zr_s-tcr_s*cr_s/2/c_slat_c_wing],\
                  lw=1,color='m')
    #------------------------------



    # Flap Lower
    #------------------------------
    if flap_type is not None:
        ax.plot([xr_f, xt_f, xt_f+ct_f, xr_f+cr_f, xr_f],
                [yr_f, yt_f, yt_f     , yr_f     , yr_f],
                [zr_f-tcr_f*cr_f/2/c_flap_c_wing ,\
                 zt_f-tct_f*ct_f/2/c_flap_c_wing ,\
                 zt_f-tct_f*ct_f/2/c_flap_c_wing ,\
                 zr_f-tcr_f*cr_f/2/c_flap_c_wing ,\
                 zr_f-tcr_f*cr_f/2/c_flap_c_wing],\
                 lw=1,color='r')

        ax.plot([ xr_f,  xt_f,  xt_f+ct_f, xr_f+cr_f, xr_f],
                [-yr_f, -yt_f, -yt_f     ,-yr_f     ,-yr_f],
                [zr_f-tcr_f*cr_f/2/c_flap_c_wing ,\
                 zt_f-tct_f*ct_f/2/c_flap_c_wing ,\
                 zt_f-tct_f*ct_f/2/c_flap_c_wing ,\
                 zr_f-tcr_f*cr_f/2/c_flap_c_wing ,\
                 zr_f-tcr_f*cr_f/2/c_flap_c_wing],\
                 lw=1,color='r')
    #------------------------------



    # Aleron Lower
    #------------------------------
    ax.plot([xr_a, xt_a, xt_a+ct_a, xr_a+cr_a, xr_a],
            [yr_a, yt_a, yt_a     , yr_a     , yr_a],
            [zr_a-tcr_a*cr_a/2/c_ail_c_wing ,\
             zt_a-tct_a*ct_a/2/c_ail_c_wing ,\
             zt_a-tct_a*ct_a/2/c_ail_c_wing ,\
             zr_a-tcr_a*cr_a/2/c_ail_c_wing ,\
             zr_a-tcr_a*cr_a/2/c_ail_c_wing],\
             lw=1,color='green')

    ax.plot([ xr_a,  xt_a,  xt_a+ct_a, xr_a+cr_a, xr_a],
            [-yr_a, -yt_a, -yt_a     ,-yr_a     ,-yr_a],
            [zr_a-tcr_a*cr_a/2/c_ail_c_wing ,\
             zt_a-tct_a*ct_a/2/c_ail_c_wing ,\
             zt_a-tct_a*ct_a/2/c_ail_c_wing ,\
             zr_a-tcr_a*cr_a/2/c_ail_c_wing ,\
             zr_a-tcr_a*cr_a/2/c_ail_c_wing],\
             lw=1,color='green')
    #------------------------------

    # Avoiding blanketing the rudder
    ax.plot([xr_h         , xr_h+b_v/np.tan(60*np.pi/180)],
            [0.0          , 0.0 ],
            [zr_h, zr_h+b_v],'k--')


    ax.plot([xr_h+cr_h         , xr_h+0.6*b_v/np.tan(30*np.pi/180)+cr_h],
            [0.0          , 0.0 ],
            [zr_h, zr_h+0.6*b_v],'k--')

    # Auxiliary landing gear lines
    if x_nlg is not None:

        # Water Spray
        ax.plot([x_nlg         , x_nlg+0.25*b_w/np.tan(22*np.pi/180)],
                [0.0          , 0.25*b_w ],
                [z_nlg, z_nlg],'k--')
    
        ax.plot([x_nlg         , x_nlg+0.25*b_w/np.tan(22*np.pi/180)],
                [0.0          , -0.25*b_w ],
                [z_nlg, z_nlg],'k--')

        # Tailstrike
        tailstrike_angle = np.arctan((-D_f/2-z_mlg)/(x_tailstrike-x_mlg))
        ax.plot([x_mlg         , L_f],
                [0.0          , 0.0 ],
                [z_mlg, z_mlg+(L_f-x_mlg)*np.tan(tailstrike_angle)],'k--')
    
        ax.plot([x_mlg         , L_f],
                [0.0          , 0.0 ],
                [z_mlg, z_mlg],'k--')


    # Create cubic bounding box to simulate equal aspect ratio
    # First create o list of possible critical points along each coordinate
    X = np.array([0, xr_w, xt_h+ct_h, xt_v+ct_v, L_f, xr_h+b_v/np.tan(60*np.pi/180), xr_h+0.6*b_v/np.tan(30*np.pi/180)+cr_h])
    Y = np.array([-yt_w, yt_w])
    Z = np.array([-D_f/2, zt_w, zt_h, zt_v, z_mlg-d_lg/2, zr_h+b_v])
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(az1, az2)

    fig.savefig(figname,dpi=300)
    
    ax.view_init(elev=90, azim=-90)
    fig.savefig(f"vista_superior.png",dpi=300)

    # Vista lateral (direita)
    ax.view_init(elev=0, azim=-90)
    fig.savefig("vista_lateral.png",dpi=300)

    # Vista frontal
    ax.view_init(elev=0, azim=0)
    fig.savefig("vista_frontal.png",dpi=300)

    plt.show()


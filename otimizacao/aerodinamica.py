import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.designTool as dt
import modules.utils as m
import numpy as np
import pprint
import matplotlib.pyplot as plt

gravity = dt.gravity
ft2m = dt.ft2m
nm2m = dt.nm2m
pi = np.pi
a = 331.3  # m/s^2

def analise_aerodinamica(airplane, show_results=True):
    
    # ------ CL - Cruise ------ #
    M = airplane['Mach_cruise']
    H = airplane['altitude_cruise']
    V = M * m.get_a(H)
    
    MTOW = airplane['W0'] # N
    rho = dt.atmosphere(H)[2] 
    CL_cruise = 0.95 * MTOW / (0.5 * rho * V**2 * airplane['S_w']) # 95% of MTOW

    if show_results:
        print('------------------------- \n')
        print(f'CL (cruise): {CL_cruise}')
        print('------------------------- \n\n')

    CD_cruise, _, dragDict = dt.aerodynamics(airplane, M, H, CL_cruise)
    if show_results: 
        print(pprint.pformat(dragDict))
        m.print_drag_table(CD_cruise, dragDict)

    # ------ CL - Landing ------ #
    MLW = airplane['MLW_frac']*airplane['W0'] # MLW - maximum landing weight
    H_landing = 0 # at sea level

    M_stall_landing, CLmax_landing = m.get_Mach_stall(airplane, MLW, 'landing', H_landing)

    M_landing = M_stall_landing*1.3
    CL_landing = CLmax_landing/(1.3**2)

    CD_landing, CLmax_landing, dragDict_landing = dt.aerodynamics(airplane, M_landing, H_landing, 
                                                                        CL_landing, 
                                                                        n_engines_failed=0, highlift_config='landing', 
                                                                        lg_down=1, h_ground=0, method=2,
                                                                        ind_drag_method='Nita',
                                                                        ind_drag_flap_method='Roskam')

    if show_results:
        print('------------------------- \n')
        print(f'CL (landing): {CL_landing}')
        print('------------------------- \n\n')

        m.print_drag_table(CD_landing, dragDict_landing)

        ## Plots
        # CD x M
        M_range = np.arange(0.6, 0.9, 0.001)
        m.plot_CD_x_M(M_range, H, CL_cruise, airplane,0.95*airplane['W0'] , '1') 

        # Drag Polar
        nome_do_pdf = "drag_polar.pdf"
        path_save = os.path.join(script_dir, nome_do_pdf)
        m.drag_polar(airplane, CL_cruise, '1', save_path=path_save)
        # TODO: checar funcao

    ## Aerodynamic Efficiency (LD)

    # LD Max
    CL_range = np.arange(-0.5,2.9,0.001)
    m.LD_max(airplane, CL_range, M, H, 0.95 * MTOW)
    LD_cruise = CL_cruise/CD_cruise
    if show_results: print(f'(L/D)_cruise = {LD_cruise:.2f} at CL = {CL_cruise:.2f}, CD = {CD_cruise:.2f}\n\n')

    # Update dictionary
    airplane['aero_CL_cruise'] = CL_cruise
    airplane['aero_CD_cruise'] = CD_cruise
    airplane['aero_CL_landing'] = CL_landing
    airplane['aero_CD_landing'] = CD_landing
    airplane['aero_CLmax_landing'] = CLmax_landing
    airplane['aero_LD_cruise'] = LD_cruise
    
    return airplane
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import designTool as dt
import matplotlib . pyplot as plt
import numpy as np
import pprint
from modules import utils as m


def analise_desempenho(airplane):
    
    W0 = airplane['W0']
    T0 = airplane['T0']

    Swvec = np.arange(50, 140, 1)   # Cria vetor de areas de asa 

    #m.plot_T0_x_Sw(airplane, Swvec, W0_guess, T0_guess)
    m.plot_T0_x_Sw(airplane, Swvec, W0, T0, op_point=(airplane['S_w'], 175927))

    sweep_wing_v = np.arange(14,32,2)*np.pi/180         # cria vetor de angulos de enflechamento (comentar essa linha caso queira plotar apenas um enflechamento)
    sweep_w = airplane['sweep_w']                       # puxa o enflechamento do nosso avião no dicionário e salva em uma variavel
    sweep_wing_v = np.append(sweep_wing_v, sweep_w)     # adiciona o enflechamento do nosso avião no vetor de enflechamentos (comentar essa linha caso queira plotar apenas um enflechamento)
    ## sweep_wing_v = np.array([sweep_w])               # plota curva para apenas o sweep angle original da aeronave 1 (descomentar essa linha caso queira plotar apenas um enflechamento)

    # slat_type_v = [None, 'leading edge flap', 'Kruger flaps','slats']               # lista com tipos de config de slat
    # flap_type_v = ['plain','single slotted','double slotted', 'triple slotted']       # lista com tipos de config de flap

    # Plot de W0 x Sw com config de flap variando para cada sweep 
    # m.plot_W0_x_Sw(airplane, Swvec, sweep_wing_v, flap_type_v, W0, T0)

    # Plot de W0 x Sw com sweep variando
    # m.plot_W0_x_sweep(airplane, Swvec, sweep_wing_v, W0, T0)

    # m.plot_T0_x_Sw(airplane, Swvec, W0, T0, op_point=(72, 211869))
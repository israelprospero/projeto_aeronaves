import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import designTool as dt
import matplotlib . pyplot as plt
import numpy as np
import pprint
from modules import utils as m

airplane = dt. standard_airplane ('my_airplane_1')
dt.geometry(airplane)
# dt.plot3d(airplane)

W0_guess = 50150*dt.gravity
T0_guess = 0.3*W0_guess 

Swvec = np.arange(50, 140, 1)   # Cria vetor de areas de asa 

### Questão 1 ###
#m.plot_T0_x_Sw(airplane, Swvec, W0_guess, T0_guess)
m.plot_T0_x_Sw(airplane, Swvec, W0_guess, T0_guess, op_point=(82, 211869))

### Questão 3 ###
### OBS: LEIA OS COMENTÁRIOS  ###

sweep_wing_v = np.arange(14,32,2)*np.pi/180         # cria vetor de angulos de enflechamento (comentar essa linha caso queira plotar apenas um enflechamento)
sweep_w = airplane['sweep_w']                       # puxa o enflechamento do nosso avião no dicionário e salva em uma variavel
sweep_wing_v = np.append(sweep_wing_v, sweep_w)     # adiciona o enflechamento do nosso avião no vetor de enflechamentos (comentar essa linha caso queira plotar apenas um enflechamento)
## sweep_wing_v = np.array([sweep_w])               # plota curva para apenas o sweep angle original da aeronave 1 (descomentar essa linha caso queira plotar apenas um enflechamento)

# slat_type_v = [None, 'leading edge flap', 'Kruger flaps','slats']               # lista com tipos de config de slat
flap_type_v = ['plain','single slotted','double slotted', 'triple slotted']       # lista com tipos de config de flap

## Plot de W0 x Sw com config de flap variando para cada sweep 
m.plot_W0_x_Sw(airplane, Swvec, sweep_wing_v, flap_type_v, W0_guess, T0_guess)

## Plot de W0 x Sw com sweep variando
m.plot_W0_x_sweep(airplane, Swvec, sweep_wing_v, W0_guess, T0_guess)

### Questão 6 ###
m.plot_T0_x_Sw(airplane, Swvec, W0_guess, T0_guess, op_point=(72, 211869))
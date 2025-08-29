from modules import designTool as dt
import matplotlib . pyplot as plt
import numpy as np
import pprint
from modules import utils as m

airplane = dt. standard_airplane ('my_airplane_1')

dt.geometry(airplane)
# dt.plot3d(airplane)

Swvec = np.arange(50, 151, 5) # Cria vetor de areas de asa 

### Questão 1 ###
m.plot_T0_x_Sw(airplane, Swvec)

### Questão 3 ###
 
sweep_wing_v = np.arange(14,32,2)*np.pi/180         # cria vetor de angulos de enflechamento
sweep_w = airplane['sweep_w']                       # puxa o enflechamento do nosso avião no dicionário e salva em uma variavel
sweep_wing_v = np.append(sweep_wing_v, sweep_w)     # adiciona o enflechamento do nosso avião no vetor de enflechamentos

# slat_type_v = [None, 'leading edge flap', 'Kruger flaps','slats']               # lista com tipos de config de slat
flap_type_v = ['plain','single slotted','double slotted', 'triple slotted']       # lista com tipos de config de flap

## Plot de W0 x Sw com config de flap variando para cada sweep 
m.plot_W0_x_Sw(airplane, Swvec, sweep_wing_v, flap_type_v)

## Plot de W0 x Sw com sweep variando
m.plot_W0_x_sweep(airplane, Swvec, sweep_wing_v)

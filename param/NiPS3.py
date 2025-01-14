"""
    source: 
        Wildes, A. R. and Stewart, J. R. and Le, M. D. and Ewings, R. A. and Rule, K. C. and Deng, G. and Anand, K.
        Magnetic dynamics of NiPS3
        Phys. Rev. B, 106:174422, Nov 2022.
    url:
        https://link.aps.org/doi/10.1103/PhysRevB.106.174422
"""

param_meV = {
    'J1':-2.6,
    'J2':0.2,
    'J3':13.6,
    'H_para':0.01,
    'H_perp':0.21
} # unit: meV

h_bar = 1.054571817e-34
gamma = 1.760859627e11 # gyromagnatic ratio
meV_2_Joule = 1.60217662e-22

NiPS3_params = {k:v*meV_2_Joule/h_bar/gamma for k,v in param_meV.items()} # unit: Tesla

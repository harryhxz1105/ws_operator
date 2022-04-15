import numpy as np
import matplotlib.pyplot as plt
import pyMMF
from Generate_theoretical_modes.functions import colorize
import sys
import threading
data_folder = "C:/project/多模/ws_operator/Generate_theoretical_modes/"

# ## Parameters
NA = 0.1
radius = 25 # in microns
areaSize = 2*radius # calculate the field on an area larger than the diameter of the fiber
n_points_modes = 100 # resolution of the window
n1 = 1.45 # index of refraction at r=0 (maximum)
wl = 1.064 # wavelength in microns
curvature = None
k0 = 2.* np.pi/wl

# solver parameters
degenerate_mode = 'sin'

# Estimate the number of modes for a graded index fiber
Nmodes_estim = pyMMF.estimateNumModesSI(wl,radius,NA,pola=1)
print(f"Estimated number of modes using the V number = {Nmodes_estim}")

# ## Simulate a graded index fiber
profile = pyMMF.IndexProfile(npoints = n_points_modes, areaSize = areaSize)
profile.initStepIndex(n1=n1,a=radius,NA=NA)

solver = pyMMF.propagationModeSolver()
solver.setIndexProfile(profile)
solver.setWL(wl)
modes = solver.solve(nmodesMax=Nmodes_estim+20,
                    mode='eig',
                    curvature = None,
                    degenerate_mode = degenerate_mode,
                    )

# 保存数据
modes.sort()
Modes = modes.getModeMatrix().T  # 模式阵

# save data
params = {}
params['NA'] = NA
params['radius'] = radius # in microns
params['areaSize'] = areaSize # calculate the field on an area larger than the diameter of the fiber
params['n_points_modes'] = n_points_modes # resolution of the window
params['n1'] = n1 # index of refraction at r=0 (maximum)
params['wl'] = wl # wavelength in microns
params['curvature'] = curvature
params['k0'] = k0
params['degenerate_mode'] = degenerate_mode
params['mode'] = 'SI'

np.savez('SI_PIM_53_0', Modes = Modes, params = params, betas = modes.betas, M = modes.m, L = modes.l)

# Show some modes
i_modes = [0,1,5,10,15,25,35]
for i in i_modes:
    Mi = Modes[i,...]
    profile = Mi.reshape([n_points_modes]*2)
    plt.figure(figsize = (4,4))
    plt.imshow(np.abs(profile))
    plt.axis('off')
    # plt.title(f'Mode {i} (l={modes.l[i]}, m={modes.m[i]})')
    # save figure
    # plt.savefig(f'mode_{i}.svg')
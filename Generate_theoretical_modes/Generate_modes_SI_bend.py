import numpy as np
import matplotlib.pyplot as plt
import pyMMF
from Generate_theoretical_modes.functions import colorize
import sys
import threading
data_folder = "C:/project/多模/fiber_speckle_analysis_python/Generate_theoretical_modes/"

## Parameters
NA = 0.1
radius = 25 # in microns
areaSize = 2*radius # calculate the field on an area larger than the diameter of the fiber
n_points = 128 # resolution of the window
n1 = 1.45 # index of refraction at r=0 (maximum)
wl = 1.064 # wavelength in microns
# curvature = None
curvature = [2e4,None]
k0 = 2.* np.pi/wl
# solver parameters
degenerate_mode = 'exp'


## Simulate a graded index fiber
# Create the fiber object
profile = pyMMF.IndexProfile(npoints = n_points, areaSize = areaSize)
# Initialize the index profile
profile.initStepIndex(n1=n1,a=radius,NA=NA)
# Instantiate the solver
solver = pyMMF.propagationModeSolver()
# Set the profile to the solver
solver.setIndexProfile(profile)
# Set the wavelength
solver.setWL(wl)
# Estimate the number of modes for a graded index fiber
Nmodes_estim = pyMMF.estimateNumModesSI(wl,radius,NA,pola=1)
print(f"Estimated number of modes using the V number = {Nmodes_estim}")


# Numerical calculations for the straight fiber
modes_straight = solver.solve(nmodesMax=Nmodes_estim+20,boundary = 'close', mode = 'eig', curvature = None, degenerate_mode=degenerate_mode)
Nmodes = sum(modes_straight.propag) # Compute for the bent fiber the same number of modes as for the straight one
modes_straight.sort()

# Numerical calculation for the bent fiber - Method 1
modes_bent = solver.solve(nmodesMax=Nmodes,boundary = 'close', mode = 'eig', curvature = curvature, degenerate_mode=degenerate_mode,)
modes_bent.sort()
betas_bent = modes_bent.betas
profiles_bent = modes_bent.getModeMatrix()
# 保存数据
Modes = profiles_bent.T  # 模式阵

# # Numerical calculation for the bent fiber - Method 2
# betas_bent2,profiles_bent2 = modes_straight.getCurvedModes(npola = 1,curvature=curvature)
# betas_bent2,profiles_bent2 = betas_bent2[:Nmodes],profiles_bent2[:,:Nmodes]


# save data
params = {}
params['NA'] = NA
params['radius'] = radius # in microns
params['areaSize'] = areaSize # calculate the field on an area larger than the diameter of the fiber
params['n_points_modes'] = n_points # resolution of the window
params['n1'] = n1 # index of refraction at r=0 (maximum)
params['wl'] = wl # wavelength in microns
params['curvature'] = curvature
params['k0'] = k0
params['degenerate_mode'] = degenerate_mode
params['mode'] = 'SI'

np.savez('SI_PIM_55_[2e4,None]', Modes = Modes, params = params, betas = modes_bent.betas, M = modes_bent.m, L = modes_bent.l)


# Show some modes
i_modes = [0,1,5,10,15,25,35]

for i in i_modes:
    Mi = Modes[i,...]
    profile = Mi.reshape([n_points]*2)
    plt.figure(figsize = (4,4))
    plt.imshow(colorize(profile,'white'))
    plt.axis('off')
    #plt.title(f'Mode {i} (l={modes.l[i]}, m={modes.m[i]})')
    # save figure
    # plt.savefig(f'mode_{i}.svg')
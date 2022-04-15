# ## Generate theoretical modes
# 
# See Supplementary Information Section 2.1
# Requires:
# * Numpy
# * Matplotlib
# * pyMMF (our custom made library to simulate multimode fiber modes available [here](https://github.com/wavefrontshaping/pyMMF)


import numpy as np
import matplotlib.pyplot as plt
import pyMMF
from Generate_theoretical_modes.functions import colorize
import sys
import threading

# ## Parameters
NA = 0.2
radius = 25 # in microns
areaSize = 2.2*radius # calculate the field on an area larger than the diameter of the fiber
n_points_modes = 128 # resolution of the window
n1 = 1.45 # index of refraction at r=0 (maximum)
wl = 1.55 # wavelength in microns
curvature = None
k0 = 2.* np.pi/wl

r_max = 3.2*radius
npoints_search = 2**8
dh = 2*radius/npoints_search

# solver parameters
change_bc_radius_step = 0.95
N_beta_coarse = 1000
degenerate_mode = 'exp'
min_radius_bc = 1.5

# ## Simulate a graded index fiber
def get_modes(npoints):
    profile = pyMMF.IndexProfile(npoints = npoints, areaSize = areaSize)
    # profile.initParabolicGRIN(n1=n1,a=radius,NA=NA)
    profile.initStepIndex(n1=n1,a=radius,NA=NA)

    solver = pyMMF.propagationModeSolver()
    solver.setIndexProfile(profile)
    solver.setWL(wl)
    modes = solver.solve(mode='radial',
                        curvature = None,
                        r_max = r_max, # max radius to calculate (and first try for large radial boundary condition)
                        dh = dh, # radial resolution during the computation
                        min_radius_bc = min_radius_bc, # min large radial boundary condition
                        change_bc_radius_step = change_bc_radius_step, #change of the large radial boundary condition if fails
                        N_beta_coarse = N_beta_coarse, # number of steps of the initial coarse scan
                        degenerate_mode = degenerate_mode,
                        )
    return modes

modes = get_modes(n_points_modes)

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

params['r_max'] = r_max
params['npoints_search'] = npoints_search
params['dh'] = dh

params['min_radius_bc'] = min_radius_bc
params['change_bc_radius_step'] = change_bc_radius_step
params['N_beta_coarse'] = N_beta_coarse
params['degenerate_mode'] = degenerate_mode
params['mode'] = 'radial'

np.savez('SI_PIM_radial_105.npz', Modes = Modes, params = params, betas = modes.betas, M = modes.m, L = modes.l)


# ## Show some modes
# i_modes = [0,1,5,10,15,25,35]

# for i in i_modes:
#     Mi = M0[...,i]
#     profile = Mi.reshape([n_points_modes]*2)
#     plt.figure(figsize = (4,4))
#     plt.imshow(colorize(profile,'white'))
#     plt.axis('off')
#     plt.title(f'Mode {i} (l={modes.l[i]}, m={modes.m[i]})')
    # save figure
    # plt.savefig(f'mode_{i}.svg')



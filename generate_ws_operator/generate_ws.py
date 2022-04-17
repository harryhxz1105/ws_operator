
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import os
from generate_Hpix import generate_H_pix
import matplotlib.image as pimg   # pimg 用于读取图片
import json
from Generate_theoretical_modes.functions import colorize, complex_correlation
cm = plt.cm.get_cmap('jet')   # 画图色盘

data_folder = "C:/project/fiber_speckle_analysis_python/Generate_theoretical_modes/"
data_folder1 = "C:/project/fiber_speckle_analysis_python/speckle/data/"
data_folder2 = "C:/project/fiber_speckle_analysis_python/speckle/data/E_in_out/"
data_folder3 = "C:/project/fiber_speckle_analysis_python/zernike/"

H_pix_0 = generate_H_pix("SI_PIM_53_0.npz")
H_pix_2e4 = generate_H_pix("SI_PIM_53_[2e4,None].npz")

# # 4.输入 pix_in * pix_in 的多幅随机散斑/一幅指定图像
# path3 = os.path.sep.join([data_folder1,'Ein_128.mat'])
# E_in = scio.loadmat(path3)['E_in']
# # for i in [1,10,20,100,500,1000,3000,4000]:
# #     plt.figure(f"E_in{i}")
# #     plt.imshow(np.abs(E_in[i, :].reshape(pix_in, pix_in)))  # 显示图片
# #     plt.axis('off')  # 不显示坐标轴
# #     plt.title(f'E_in{i}')
# #     plt.savefig(data_folder2 + f"/E_in{i}.png")
#
# # 输入校徽图案
# # E_in = pimg.imread('logo.png')  # 读取和代码处于同一目录下的 logo.png
# # def rgb2gray(rgb):
# #     return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
# # E_in = rgb2gray(logo_in).reshape(-1,1)
#
# # 5.计算输出散斑
# # 输出散斑电场
# E_out = np.matmul(E_in, H_pix)
# # np.savez(data_folder1 + '/E_out.npz', E_out = E_out)
#
# # 输出散斑光强
# E_out_intensity = np.abs(E_out)**2
# np.savez(data_folder1 + '/E_out_intensity.npz', E_out_intensity = E_out_intensity)
#
# for i in [1,10,20,30,50,70,90,100,200,300,400,500,1000,2000,3000,4000]:
#     plt.figure(f"E_out{i}")
#     plt.imshow(E_out_intensity[i, :].reshape(pix_out, pix_out))  # 显示输出散斑
#     plt.axis('off')
#     plt.title(f'E_out{i}')
#     plt.savefig(data_folder2 + f"/E_out{i}.png")









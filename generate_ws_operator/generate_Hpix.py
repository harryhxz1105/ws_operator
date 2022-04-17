import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import os
from functions import SNR, gen_gaussian_noise
import matplotlib
# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

cm = plt.cm.get_cmap('jet')   # 画图色盘
data_folder1 = r"F:\project\多模\ws_operator\Generate_theoretical_modes"


def generate_H_pix(modes_file,):
    "----------------------1.导入模式矩阵  N（mode）*N（pix）每一行是一个模式----------------------"
    # path1 = os.path.sep.join([data_folder1, 'SI_PIM_53_0.npz'])
    path1 = os.path.sep.join([data_folder1, f"{modes_file}"])
    data = np.load(path1)
    M_in = data.f.Modes
    # 光纤支持的模式数
    nmodes = M_in.shape[0]
    # 光纤模式矩阵归一化处理
    for i in range(0, nmodes):
        M_in[i, :] = (M_in[i, :]) / np.sqrt(np.sum(np.abs(M_in[i, :]) ** 2))
    # number of in pixels
    pix_in = np.int32(np.sqrt(M_in.shape[1]))

    # M_out = data.f.Modes
    # # 光纤模式矩阵归一化处理
    # for i in range(0, nmodes):
    #     M_out[i, :] = (M_out[i, :])/np.sqrt(np.sum(np.abs(M_out[i, :])**2))
    # # number of out pixels
    # pix_out = np.int32(np.sqrt(M_out.shape[1]))

    M_out = M_in
    pix_out = pix_in

    "-------------------2.生成一个对角矩阵（Nmodes*Nmodes）,模拟 模式域传输矩阵H_mode---------------"
    # # method 1
    # # 光纤模式的索引（m角向数；l径向数）
    # M = data.f.M
    # L = data.f.L
    # H_mode = np.identity(nmodes)
    # # 2|m|+l相同的模式之间存在耦合
    # for i in range(0, len(M)):
    #     for j in range(i, len(M)):
    #         if 2*np.abs(M[j])+L[j] == 2*np.abs(M[i])+L[i]:
    #             H_mode[i, j] = np.random.rand()*np.exp(1j*2*np.pi*np.random.rand())
    #             H_mode[j, i] = np.random.rand()*np.exp(1j*2*np.pi*np.random.rand())

    # # method 2 只在主对角线上有元素
    H_mode = np.identity(nmodes)
    for i in range(0, nmodes):
        H_mode[i, i] = np.random.rand() * np.exp(1j * 2 * np.pi * np.random.rand())

    plt.figure("H_mode")
    plt.imshow(np.abs(H_mode), cmap=cm)
    plt.axis('off')
    plt.title('H_mode')
    # plt.savefig(data_folder2 + "/H_mode.png")

    # 3.逆推 像素域传输矩阵
    H1 = np.matmul(H_mode, M_in)
    H_pix = np.matmul(np.conj(M_out).T, H1)
    # np.savez(data_folder1 + 'H_pix', H_pix = H_pix)

    # 矩阵过大时需分块存储
    # H_pix_1 = np.matmul(np.conj(M_out).T,H1[:,:np.int32((M_in.shape[1])/2)])
    # H_pix_2 = np.matmul(np.conj(M_out).T,H1[:,np.int32((M_in.shape[1])/2):])

    plt.figure("H_pix")
    plt.imshow(np.abs(H_pix), cmap=cm)
    plt.axis('off')
    plt.title('H_pix')
    # plt.savefig(data_folder2 + "/H_pix.png")

    return H_pix



#
# # 1.①读取输入散斑矩阵，并计算奇异值/②直接读取奇异值
# # --------------------①-------------------------
# # path1 = os.path.sep.join([data_folder1,'A128.mat'])
# # A = scio.loadmat(path1)['A_big']
# # u, s, v = np.linalg.svd(A, full_matrices=0)
# # np.savez(data_folder1 + '/A_svd.npz', u=u ,s=s ,v=v)
# # --------------------②-------------------------
# path2 = os.path.sep.join([data_folder1,'A_svd.npz'])
# data = np.load(path2)
# u = data.f.u
# s = data.f.s
# v = data.f.v
#
# # 2.透射型物体
# path3 = os.path.sep.join([data_folder1,'Img5.mat'])
# img_real = scio.loadmat(path3)['Img']
# img_real[img_real<50]=0
# img_real[img_real>0]=1
# img_real = np.transpose(img_real)
#
# plt.figure("x_TO")
# plt.imshow(np.transpose(np.abs(img_real)), cmap='gray')
# plt.axis('off')
# plt.title('x_TO')
#
#
# # 3.孔探测器单像素测量值矩阵
# path3 = os.path.sep.join([data_folder1,'y1285.mat'])
# y = scio.loadmat(path3)['y']
# y_initial = y
#
# # Y添加高斯噪声
# snr_y = 50  # 信噪比：dB
# e_y = gen_gaussian_noise(y, snr_y)
# # e_y = 1*np.linalg.norm(y)/np.sqrt(len(y))*10**(-snr_y/20)*np.random.randn(*y.shape)
# y = y + e_y
# print(f"SNR_y={SNR(y_initial, y)}")
#
# # plt.figure('e_y')
# # plt.hist(e_y, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
# # # 显示横轴标签
# # plt.xlabel("区间")
# # # 显示纵轴标签
# # plt.ylabel("频数/频率")
# # # 显示图标题
# # plt.title("e_y频数/频率分布直方图")
# # plt.show()
#
#
#
# # TSVD恢复算法，重构图像
# # -----------TSVD--迭代进行k（1-5000）次截断--------------
# pearson = [] # 皮尔逊系数
# snr = []
# x_recover_record = [] # 存储不同k截断时，对应的重构图像
# sum = 0
# for k in range(1, 5001):
#     print(k)
#     x_rank = (u[:, k - 1:k].T @ y @ v[k - 1:k, :]) / s[k - 1]
#     # x_rank = (u[:, k-1:k].T@y@v[k-1:k, :])/s[k-1] + (u[:, k-1:k].T@y@e_y[k-1:k, :])/s[k-1]
#
#     sum = sum + x_rank
#     sum[sum<0]=0
#
#     snr1 = SNR(np.abs(img_real.reshape(1, -1)), np.abs(sum))
#     # pearson1 =np.abs(np.corrcoef(np.abs(img_real.reshape(1, -1)), np.abs(sum))[0][1])
#
#     x_recover_record.append(sum)
#     # pearson.append(pearson1)
#     snr.append(snr1)
#
#
# plt.figure("k-SNR")
# plt.plot(snr)
# plt.title('k-SNR')
#
# plt.figure("x_recovery")
# plt.imshow(np.transpose(np.abs(x_recover_record[400]).reshape(100,100)), cmap='gray')
# plt.axis('off')
# plt.title('x_recovery')
#
#
#
#
# # -----------TSVD 方法二 --------------
# # s1 = 1/s
# # pearson = [] # 皮尔逊系数
# # x_recover_record = [] # 存储不同k截断时，对应的重构图像
# # # 遍历k(1-5000)次奇异值截断
# # for k in range(1, 5001):
# #     u_k = u[:, 0:k]
# #     s2_k = np.diag(s1[0:k])
# #     v_k = v[0:k, :]
# #
# #     A_inverse = np.transpose(u_k @ s2_k @ v_k)
# #     x_recover = A_inverse @ y
# #     x_recover[x_recover<0]=0
# #
# #     pearson1 = np.abs(np.corrcoef(np.abs(img_real.reshape(-1, 1)), np.abs(x_recover))[0][1])
# #     img_recover = x_recover.reshape(100, 100)
# #
# #     pearson.append(pearson1)
# #     x_recover_record.append(x_recover)
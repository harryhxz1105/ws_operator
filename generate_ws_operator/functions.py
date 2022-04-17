
import numpy as np
import matplotlib.pyplot as plt


def SNR(clean_img,noise_img):
    '''
    计算信噪比
    '''
    # 归一化
    clean_img = clean_img/np.sum(clean_img)
    noise_img = noise_img/np.sum(noise_img)

    # 原始信号
    clean_signal=clean_img
    signal = np.mean(clean_signal)
    # 计算噪声
    noise_signal=noise_img-clean_img
    noise = np.sqrt(np.mean(noise_signal**2))

    # 计算信噪比
    snr=20*np.log10(signal/noise)

    return snr


def gen_gaussian_noise(signal,snr):
    """
    加入高斯白噪声 Additive White Gaussian Noise
    :param signal: 原始信号
    :param snr: 添加噪声的信噪比
    :return: 噪声信号
    """
    noise = np.random.randn(*signal.shape) # *signal.shape 获取样本序列的尺寸
    noise = noise - np.mean(noise)
    # plt.figure('noise')
    # plt.hist(noise, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.show

    y = np.array(signal, dtype='int64')
    signal_power = np.sum(y ** 2) / y.shape[0]
    noise_variance = signal_power / (10 ** (snr / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise  # np.std 标准差
    # noise = np.sqrt(noise_variance  * noise

    return noise
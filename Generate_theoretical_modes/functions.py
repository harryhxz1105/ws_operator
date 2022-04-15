from colorsys import hls_to_rgb
import numpy as np

def colorize(z, theme = 'dark', saturation = 1., beta = 1.4, transparent = False, alpha = 1.):
    '''
    幅度和相位画在一幅图中
    :param z:
    :param theme:
    :param saturation: 饱和度
    :param beta:
    :param transparent: 背景透明
    :param alpha:
    :return:
    '''
    r = np.abs(z)
    r /= np.max(np.abs(r))  # 幅度归一化处理
    arg = np.angle(z)  # 相位

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1./(1. + r**beta) if theme == 'white' else 1.- 1./(1. + r**beta)
    s = saturation # 饱和度

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    if transparent:
        a = 1.-np.sum(c**2, axis = -1)/3
        alpha_channel = a[...,None]**alpha
        return np.concatenate([c,alpha_channel], axis = -1)
    else:
        return c
    
    
def complex_correlation(Y1,Y2):
    Y1 = Y1-Y1.mean()
    Y2 = Y2-Y2.mean()
    return np.abs(np.sum(Y1.ravel() * Y2.ravel().conj())) \
           / np.sqrt(np.sum(np.abs(Y1.ravel())**2) *np.sum(np.abs(Y2.ravel())**2))


tr = lambda A,B: np.trace(np.abs(A@B.transpose().conjugate())**2)

fidelity = lambda A,B: tr(A,B)/(np.sqrt(tr(A,A)*tr(B,B)))
import numpy as np
from PIL import Image, ImageOps
import math
import random
import sys, getopt
from scipy.fft import fft2, fftfreq, fftshift, ifft2
from scipy import fftpack, ndimage
from matplotlib import pyplot as plt
import cmath
from scipy.signal import convolve2d
from matplotlib.pyplot import imread
import os
import time

def M(x, y):
    if(x**2 + y**2 < (math.pi/2)**2):
        return 1
    else:
        return 0

def p_s(x, y, delta_z):
    return 0.375 * delta_z * math.pi * (x**2 + y**2)

    
def p(x, y):
    return 0.5 * math.pi * (x**2 + y**2)

def h(delta_z):  # создает матрицу для фильтра нужной нам размерности (h)
# от сигма зависит степень размытия. чем больше сигма тем больше размытие
    
    matr = np.zeros(shape = (512,512), dtype = np.complex128) # матрица из 0 (по размеру изборжания подаваемого)
                                          
    #|F(M(x,y)∙exp{i∙p_s (x,y,∆z_nm )+i∙p(x,y)})|
    
    #p_s = 0.375* math.pi * (разность между слоями) * (x**2 + y**2) 
     
    for i in range(matr.shape[0]):
        for j in range(matr.shape[1]):
            x = math.pi/511*(i-255)
            y = math.pi/511*(j-255)
            matr[i, j] = M(x,y) * cmath.exp(1j * p_s(x, y, delta_z) + 1j * p(x, y))
            
    return matr

im = Image.open("src_0000.png",'r')
# im = im.convert('1')
data = np.array(im)                         #представляем как массив
data_im_fl = data[:,:,0].astype(np.float)
print(data_im_fl.shape)
im_f_n = np.zeros(shape = (1024,1024), dtype = float)
im_f_n[0:512,0:512] = data_im_fl            #расширяем
# # im_f_n[0:2048,0:2048] = data_im_fl
# print(im_f_n.shape)
# np.savetxt("src_00001.txt", im_f_n, fmt='%.18e')
np.savetxt("src_0000.txt", im_f_n, fmt='%i')


# h_f_n = np.zeros(shape = (1024,1024), dtype = float)
h_f_n = np.zeros(shape = (512,512), dtype = float)
h_f_n[0:512, 0:512] = h(delta_z = math.pi * 20)
print(h_f_n.max())
print(h_f_n.min())
np.savetxt("h_func.txt", h_f_n, fmt='%f')

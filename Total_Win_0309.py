#%%


#%%
#for imresize

from __future__ import print_function


from math import ceil, floor

def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape

def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale

def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x>=-1),x<0)
    greaterthanzero = np.logical_and((x<=1),x>=0)
    f = np.multiply((x+1),lessthanzero) + np.multiply((1-x),greaterthanzero)
    return f

def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f

def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1 # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1) # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices

def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)        
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg =  np.sum(weights*((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg =  np.sum(weights*((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out

def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method is 'bicubic':
        kernel = cubic
    elif method is 'bilinear':
        kernel = triangle
    else:
        print ('Error: Unidentified method supplied')
        
    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        print ('Error: scalar_scale OR output_shape should be defined!')
        return
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I) 
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B


#im2double

def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max 


def im2doublef(im):
    info = np.finfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max 


#######################################################################################################################################
#%%


#for Rendering
import numpy as np

import math
from math import pi

import scipy.io
import scipy.ndimage

import cv2 

import matplotlib.image as Img

import PIL
from PIL import Image, ImageMath


#for network.py

import torch
import time

#for run.py

import os
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.serialization    # it was removed in torch v1.0.0 or higher version.
#from Network import TOFlow
import matplotlib.pyplot as plt
import sys
import getopt
from scipy import io


#전처리기





class Params:
    LFU_W = 50
    
    
    HEIGHT = 2048
    WIDTH  = 4096
    RESIZE_HEIGHT = HEIGHT/4
    RESIZE_WIDTH = WIDTH/4
    
    """
    #size down for GPU
    HEIGHT = 2048/4
    WIDTH  = 4096/4
    RESIZE_HEIGHT = HEIGHT
    RESIZE_WIDTH = WIDTH
    """

    Search_term = 20
    Filter_size = 200
    Filter_fov  = pi/24

    Filter_fov = round(Filter_fov*10000)
    Filter_fov = Filter_fov/10000

    Search_bound = 50



#% Functions
def SetParams():

    Params.LFU_W = 50
    """
    Params.HEIGHT = 2048
    Params.WIDTH  = 4096

    Params.RESIZE_HEIGHT = Params.HEIGHT/4
    Params.RESIZE_WIDTH = Params.WIDTH/4
    """

    #size down for GPU
    Params.HEIGHT = 512
    Params.WIDTH  = 1024

    Params.RESIZE_HEIGHT = Params.HEIGHT
    Params.RESIZE_WIDTH = Params.WIDTH

    Params.Search_term = 20
    Params.Filter_size = 200
    Params.Filter_fov  = pi/24
    Params.Search_bound = 50

Params.Filter_fov = round(Params.Filter_fov*10000)
Params.Filter_fov = Params.Filter_fov/10000

###로드

def LoadLFs5K():

    print("LoadLFs5K...\n")

    x = np.arange(Params.HEIGHT)
    y = np.arange(Params.WIDTH)
    TY, TX = np.meshgrid(x,y)
    TX = TX.reshape(-1,1) 
    TY = TY.reshape(-1,1) 

    LF = [[],[]]

    for i in range(0,len(LF)):
        LF[i] = np.zeros(3 * Params.LFU_W * Params.HEIGHT * Params.WIDTH * 3 , 'uint8')
        for n in range(1,Params.LFU_W * 3+1):
            if (i == 0):
                file = '%s/00/%04d.jpeg' %(Params.cur_dir, n) 
            else:
                file = '%s/02/%04d.jpeg' %(Params.cur_dir, n) 


            #img = Img.imread(file)
            #for GPU
            img = imresize(Img.imread(file), output_shape=(Params.HEIGHT, Params.WIDTH))
            """
            img = Img.imread(file)
            img = imresize(img)
            """
            img_C1 = img[:,:,0]
            img_C2 = img[:,:,1]
            img_C3 = img[:,:,2]
            nn = n - 1


            img_C1 = img_C1.T
            img_C1 = img_C1.reshape(-1,1)

            img_C2 = img_C2.T
            img_C2 = img_C2.reshape(-1,1)

            img_C3 = img_C3.T
            img_C3 = img_C3.reshape(-1,1) 
            



            TX =TX.astype('int64')
            TY =TY.astype('int64')
            
    
            LF[i][(nn) * (Params.HEIGHT * Params.WIDTH * 3) + (TX) * (Params.HEIGHT * 3) + (TY) * 3 + 3 -1] = img_C1[:]
            LF[i][(nn) * (Params.HEIGHT * Params.WIDTH * 3) + (TX) * (Params.HEIGHT * 3) + (TY) * 3 + 2 -1] = img_C2[:]
            LF[i][(nn) * (Params.HEIGHT * Params.WIDTH * 3) + (TX) * (Params.HEIGHT * 3) + (TY) * 3 + 1 -1] = img_C3[:]
    
    

            print(['LF',i, ' - ', n, ', ',nn])
        

    return LF





#%%

#@mfunction("IMAGE_MAT")
def inter8_mat5K(LF=None, P_r=None, P_1=None, P_2=None, U_r=None, U_1=None, U_2=None, H_r=None, H_1=None, H_2=None, c=None):

    print("inter8_mat5K...\n")

    height = Params.HEIGHT
    width = Params.WIDTH

    P_1[P_r == 1] = 0
    P_2[P_r == 0] = 0 #print P_2

    U_1[U_r == 1] = 0
    U_2[U_r == 0] = 0 #print U_2

    H_1[H_r == 1] = 0
    H_2[H_r == 0] = 0 #print H_2

    #arrays used as indices must be of integer (or boolean) type, unt64 for long data
    P_1 = np.array(P_1,dtype=np.int64)
    P_2 = np.array(P_2,dtype=np.int64)
    U_1 = np.array(U_1,dtype=np.int64)
    U_2 = np.array(U_2,dtype=np.int64)
    H_1 = np.array(H_1,dtype=np.int64)
    H_2 = np.array(H_2,dtype=np.int64)

    if(c == 1):
        IMAGE_MAT = ((1.0 - P_r) * \
                ((1.0 - U_r) * ((1.0 - H_r) * im2double(LF[(P_1) * (height * width * 3) + U_1 * (height * 3) + H_1 * 3 + 1 -1]) + \
                                                    H_r * im2double(LF[(P_1) * (height * width * 3) + U_1 * (height * 3) + H_2 * 3 + 1 -1])) + \
                         ((U_r) * ((1.0 - H_r) * im2double(LF[(P_1) * (height * width * 3) + U_2 * (height * 3) + H_1 * 3 + 1 -1]) + \
                                                    H_r * im2double(LF[(P_1) * (height * width * 3) + U_2 * (height * 3) + H_2 * 3 + 1 -1]))))) + \
                        ((P_r) * \
                ((1.0 - U_r) * ((1.0 - H_r) * im2double(LF[(P_2) * (height * width * 3) + U_1 * (height * 3) + H_1 * 3 + 1 -1]) + \
                                       H_r * im2double(LF[(P_2) * (height * width * 3) + U_1 * (height * 3) + H_2 * 3 + 1 -1])) + \
                      ((U_r) * ((1.0 - H_r) * im2double(LF[(P_2) * (height * width * 3) + U_2 * (height * 3) + H_1 * 3 + 1 -1]) + \
                                       H_r * im2double(LF[(P_2) * (height * width * 3) + U_2 * (height * 3) + H_2 * 3 + 1 -1])))))
    elif(c == 2):
        IMAGE_MAT = ((1.0 - P_r) * \
                ((1.0 - U_r) * ((1.0 - H_r) * im2double(LF[(P_1) * (height * width * 3) + U_1 * (height * 3) + H_1 * 3 + 2 -1]) + \
                                       H_r * im2double(LF[(P_1) * (height * width * 3) + U_1 * (height * 3) + H_2 * 3 + 2 -1])) + \
                      ((U_r) * ((1.0 - H_r) * im2double(LF[(P_1) * (height * width * 3) + U_2 * (height * 3) + H_1 * 3 + 2 -1]) + \
                                       H_r * im2double(LF[(P_1) * (height * width * 3) + U_2 * (height * 3) + H_2 * 3 + 2 -1]))))) + \
                        ((P_r) * \
                ((1.0 - U_r) * ((1.0 - H_r) * im2double(LF[(P_2) * (height * width * 3) + U_1 * (height * 3) + H_1 * 3 + 2 -1]) + \
                                       H_r * im2double(LF[(P_2) * (height * width * 3) + U_1 * (height * 3) + H_2 * 3 + 2 -1])) + \
                      ((U_r) * ((1.0 - H_r) * im2double(LF[(P_2) * (height * width * 3) + U_2 * (height * 3) + H_1 * 3 + 2 -1]) + \
                                       H_r * im2double(LF[(P_2) * (height * width * 3) + U_2 * (height * 3) + H_2 * 3 + 2 -1])))))
    elif(c == 3):
        IMAGE_MAT = ((1.0 - P_r) * \
                ((1.0 - U_r) * ((1.0 - H_r) * im2double(LF[(P_1) * (height * width * 3) + U_1 * (height * 3) + H_1 * 3 + 3 -1]) + \
                                       H_r * im2double(LF[(P_1) * (height * width * 3) + U_1 * (height * 3) + H_2 * 3 + 3 -1])) + \
                      ((U_r) * ((1.0 - H_r) * im2double(LF[(P_1) * (height * width * 3) + U_2 * (height * 3) + H_1 * 3 + 3 -1]) + \
                                       H_r * im2double(LF[(P_1) * (height * width * 3) + U_2 * (height * 3) + H_2 * 3 + 3 -1]))))) + \
                        ((P_r) * \
                ((1.0 - U_r) * ((1.0 - H_r) * im2double(LF[(P_2) * (height * width * 3) + U_1 * (height * 3) + H_1 * 3 + 3 -1]) + \
                                       H_r * im2double(LF[(P_2) * (height * width * 3) + U_1 * (height * 3) + H_2 * 3 + 3 -1])) + \
                      ((U_r) * ((1.0 - H_r) * im2double(LF[(P_2) * (height * width * 3) + U_2 * (height * 3) + H_1 * 3 + 3 -1]) + \
                                       H_r * im2double(LF[(P_2) * (height * width * 3) + U_2 * (height * 3) + H_2 * 3 + 3 -1])))))

    print()
    
    IMAGE_MAT = np.rint((IMAGE_MAT*10000))
    IMAGE_MAT = IMAGE_MAT/10000

    IMAGE_MAT = IMAGE_MAT * 255
    IMAGE_MAT = np.rint((IMAGE_MAT))

    return IMAGE_MAT


def inter8_mat_flow5K(LF=None, P_r=None, P_1=None, P_2=None, U_r=None, U_1=None, U_2=None, H_r=None, H_1=None, H_2=None):

    print("inter8_mat_flow5K")

    height = Params.HEIGHT
    width = Params.WIDTH

    P_1[P_r == 1] = 0
    P_2[P_r == 0] = 0 #print P_2

    U_1[U_r == 1] = 0
    U_2[U_r == 0] = 0 #print U_2

    H_1[H_r == 1] = 0
    H_2[H_r == 0] = 0 #print H_2


    #데이터 타입 에러 수정위함
    P_1 = np.array(P_1,dtype=np.int64)
    P_2 = np.array(P_2,dtype=np.int64)
    U_1 = np.array(U_1,dtype=np.int64)
    U_2 = np.array(U_2,dtype=np.int64)
    H_1 = np.array(H_1,dtype=np.int64)
    H_2 = np.array(H_2,dtype=np.int64)
    

    FLOW_MAT = ((1.0 - P_r) * \
               ((1.0 - U_r) * ((1.0 - H_r) * LF[(P_1) * (height * width) + U_1 * height + H_1 + 1 -1] + \
                                       H_r  * LF[(P_1) * (height * width) + U_1 * height + H_2 + 1 -1]) + \
                     ((U_r) * ((1.0 - H_r) * LF[(P_1) * (height * width) + U_2 * height + H_1 + 1 -1] + \
                                       H_r  * LF[(P_1) * (height * width) + U_2 * height + H_2 + 1 -1])))) + \
                     ((P_r) * \
               ((1.0 - U_r) * ((1.0 - H_r) * LF[(P_2) * (height * width) + U_1 * height + H_1 + 1 -1] + \
                                       H_r  * LF[(P_2) * (height * width) + U_1 * height + H_2 + 1 -1]) + \
                     ((U_r) * ((1.0 - H_r) * LF[(P_2) * (height * width) + U_2 * height + H_1 + 1 -1] + \
                                       H_r  * LF[(P_2) * (height * width) + U_2 * height + H_2 + 1 -1]))))
      

    FLOW_MAT = np.ceil((FLOW_MAT*10000))
    FLOW_MAT= FLOW_MAT/10000

    return FLOW_MAT






#%%


###렌더링


def RenderingUserViewLF_AllinOne5K(LF=None, LFDisparity=None, FB=None, viewpoint=None, DIR=None):
    

    sphereW = Params.WIDTH
    sphereH = Params.HEIGHT

    CENTERx = viewpoint.lon
    CENTERy = viewpoint.lat

    # output view is 3:4 ratio
    new_imgW = np.floor(viewpoint.diag * 4 / 5 + 0.5)
    new_imgH = np.floor(viewpoint.diag * 3 / 5 + 0.5)

    new_imgW = int(new_imgW)
    new_imgH = int(new_imgH)

    OutView = np.zeros((new_imgH, new_imgW, 3))
    TYwarp,TXwarp = np.mgrid[0:new_imgH, 0:new_imgW]

    TX = TXwarp
    TY = TYwarp
    TX = (TX - 0.5 - new_imgW/2)
    TY = (TY - 0.5 - new_imgH/2)

    #의심
    
    TX = TX +1
    TY = TY +1
    

    r = (viewpoint.diag/2) / np.tan(viewpoint.fov/2)
    R = np.sqrt(TY ** 2 + r ** 2) 
    # Calculate LF_n
    ANGy = np.arctan(-TY / r)
    ANGy = ANGy + CENTERy



    if (FB == 1):
        ANGn = np.cos(ANGy) * np.arctan(TX / r)
        ANGn = ANGn + CENTERx
        Pn = (Params.LFU_W / 2 - viewpoint.pos_y) * np.tan(ANGn) + viewpoint.pos_x + (3 * Params.LFU_W / 2)
    elif (FB == 2):
        ANGn = np.cos(ANGy) * np.arctan(-TX / r) 
        ANGn = ANGn - CENTERx
        Pn = (Params.LFU_W / 2 + viewpoint.pos_y) * np.tan(ANGn) + viewpoint.pos_x + (3 * Params.LFU_W / 2)

    X = np.sin(ANGy) * R
    Y = -np.cos(ANGy) * R
    Z = TX

    ANGx = np.arctan2(Z, -Y)
    RZY = np.sqrt(Z ** 2 + Y ** 2)  
    ANGy = np.arctan(X / RZY)#or ANGy = atan2(X, RZY); 

    RATIO = 1
    ANGy = ANGy * RATIO

    ANGx = ANGx + CENTERx

    ANGx[abs(ANGy) > pi / 2] = ANGx[abs(ANGy) > pi / 2] + pi
    ANGx[ANGx > pi] = ANGx[ANGx > pi] - 2 * pi

    ANGy[ANGy > pi / 2] = pi / 2 - (ANGy[ANGy > pi / 2] - pi / 2)
    ANGy[ANGy < -pi / 2] = -pi / 2 + (ANGy[ANGy < -pi / 2] + pi / 2)

    Px = (ANGx + pi) / (2 * pi) * sphereW + 0.5
    Py = ((-ANGy) + pi / 2) / pi * sphereH + 0.5

    if (DIR == 2):
        Px = Px + Params.WIDTH / 4
    elif (DIR == 3):
        Px = Px + Params.WIDTH / 2
    elif (DIR == 4):
        Px = Px - Params.WIDTH / 4

    Px[Px < 1] = Px[Px < 1] + Params.WIDTH
    Px[Px > Params.WIDTH] = Px[Px > Params.WIDTH] - Params.WIDTH

    INDxx = np.argwhere(Px < 1)
    Px[INDxx]= Px[INDxx] + sphereW

    Pn0 = np.floor(Pn)
    Pn1 = np.ceil(Pn)
    Pnr = Pn - Pn0


    Px0 = np.floor(Px)
    Px1 = np.ceil(Px)
    Pxr = Px - Px0

    Py0 = np.floor(Py)
    Py1 = np.ceil(Py)
    Pyr = Py - Py0


    Pnr = np.rint((Pnr*10000))
    Pnr = Pnr/10000

    Pxr = np.rint((Pxr*10000))
    Pxr = Pxr/10000

    Pyr = np.rint((Pyr*10000))
    Pyr = Pyr/10000

    #210->012 rgb 
    #cv2 사용 안하면 그대로 
    OutView[:, :, 2] = inter8_mat5K(LF, Pnr, Pn0, Pn1, Pxr, Px0, Px1, Pyr, Py0, Py1, 1)
    OutView[:, :, 1] = inter8_mat5K(LF, Pnr, Pn0, Pn1, Pxr, Px0, Px1, Pyr, Py0, Py1, 2)
    OutView[:, :, 0] = inter8_mat5K(LF, Pnr, Pn0, Pn1, Pxr, Px0, Px1, Pyr, Py0, Py1, 3)
    OutFlow = inter8_mat_flow5K(LFDisparity, Pnr, Pn0, Pn1, Pxr, Px0, Px1, Pyr, Py0, Py1)
  


    Py = np.pad(Py,[(1,1),(0,0)], mode='edge')
    Py = np.ceil((Py*10000))
    Py = Py/10000

    My = 2 / (Py[2:np.size(Py, 0), :] - Py[0:(np.size(Py, 0)-2), :])

    My[0,:] = My[0,:] / 2

    My[np.size(My, 0)-1, :] = My[np.size(My, 0)-1, :] / 2


    OutFlow = My * OutFlow 

    return OutView, OutFlow





#%%


#clear(mstring('clc'))
print ("\n"*80)

print( pi, "\n")

print("start\n")

SetParams()

sample = 'S3'
mode = 'DF'

out_forward = 'M3DLF_%s/FlowForward_%s.mat' %(sample, mode) 
out_backward = 'M3DLF_%s/FlowBackward_%s.mat' %(sample, mode) 

Params.K = 0.1
Params.cur_dir = 'M3DLF_'+sample+'/Input'
Params.out_dir = 'M3DLF_'+sample+'/Output'

class viewpoint:
    pos_x = 0
    pos_y = 0
    lon = 0
    lat = 0
    diag = 0
    fov = 0

viewpoint.pos_x = 0
viewpoint.pos_y = 0
viewpoint.lon = np.deg2rad(0)
viewpoint.lat = np.deg2rad(0)
viewpoint.diag = 860
viewpoint.fov = pi / 3

viewpoint.fov = round(viewpoint.fov*10000)
viewpoint.fov = viewpoint.fov/10000

LFDisp_forward = np.zeros(3 * Params.LFU_W * Params.HEIGHT * Params.WIDTH)
LFDisp_backward = np.zeros(3 * Params.LFU_W * Params.HEIGHT * Params.WIDTH)


x = np.arange(Params.HEIGHT)
y = np.arange(Params.WIDTH)
TY ,TX = np.meshgrid(x,y)


TX = TX.reshape(-1,1) 
TY = TY.reshape(-1,1) 

mat_file = scipy.io.loadmat(out_forward)
LFDisparity_forward = mat_file['LFDisparity_forward']

for n in range(0,Params.LFU_W * 3): 
    nn = n - 1
    imageflow = np.transpose(np.expand_dims(LFDisparity_forward[n,:,:],axis=0), (1,2,0)) 
    imageflow = np.squeeze(imageflow, axis=2)
    imageflow = imresize(imageflow,output_shape=(Params.HEIGHT, Params.WIDTH))
    imageflow = (Params.HEIGHT / Params.RESIZE_HEIGHT) * imageflow
    imageflow = imageflow.T
    imageflow = imageflow.reshape(-1,1)
    LFDisp_forward[(nn+1) * (Params.HEIGHT * Params.WIDTH) + (TX) * (Params.HEIGHT) + (TY)] = imageflow 
    print(str(n)) 

print("Loading...\n")

LFDisp_forward = np.rint((LFDisp_forward*10000))
LFDisp_forward = LFDisp_forward/10000

mat_file = scipy.io.loadmat(out_backward)
LFDisparity_backward = mat_file['LFDisparity_backward']

for n in range(0,Params.LFU_W * 3): 
    nn = n - 1
    imageflow = np.transpose(np.expand_dims(LFDisparity_backward[n,:,:],axis=0), (1,2,0)) 
    imageflow = np.squeeze(imageflow, axis=2)
    imageflow = imresize(imageflow,output_shape=(Params.HEIGHT, Params.WIDTH))
    imageflow = (Params.HEIGHT / Params.RESIZE_HEIGHT) * imageflow
    imageflow = imageflow.T
    imageflow = imageflow.reshape(-1,1)
    LFDisp_backward[(nn+1) * (Params.HEIGHT * Params.WIDTH) + (TX) * (Params.HEIGHT) + (TY)] = imageflow 
    print(str(n)) 

LFDisp_backward = np.rint((LFDisp_backward*10000))
LFDisp_backward = LFDisp_backward/10000

print("Loading...\n")

# load color data
LF = LoadLFs5K()

print("Loading end...\n")

#%%

#% One view rendering (test)

viewdirection = 0
viewpoint.lon = np.deg2rad(0)
viewpoint.lat = np.deg2rad(0)
viewpoint.pos_x = 0 
viewpoint.pos_y = 0 
if (viewdirection == 0):
    viewpoint.lon = viewpoint.lon
elif (viewdirection == 1):
    viewpoint.lon = viewpoint.lon + 180



OutViewF, OutFlowF = RenderingUserViewLF_AllinOne5K(LF[0], LFDisp_forward, 1, viewpoint, 1)
OutFlowF = np.rint((OutFlowF*10000))
OutFlowF = OutFlowF/10000

OutViewB, OutFlowB = RenderingUserViewLF_AllinOne5K(LF[1], LFDisp_backward, 2, viewpoint, 1)
OutFlowB = np.rint((OutFlowB*10000))
OutFlowB = OutFlowB/10000


OutViewF = np.uint8(OutViewF)
OutViewB = np.uint8(OutViewB)


# %%

#network.py

arguments_strModel = 'sintel-final'
SpyNet_model_dir = './models'  # The directory of SpyNet's weights

def normalize(tensorInput):
    tensorRed = (tensorInput[:, 0:1, :, :] - 0.485) / 0.229
    tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
    tensorBlue = (tensorInput[:, 2:3, :, :] - 0.406) / 0.225
    return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)

def denormalize(tensorInput):
    tensorRed = (tensorInput[:, 0:1, :, :] * 0.229) + 0.485
    tensorGreen = (tensorInput[:, 1:2, :, :] * 0.224) + 0.456
    tensorBlue = (tensorInput[:, 2:3, :, :] * 0.225) + 0.406
    return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)

Backward_tensorGrid = {}

def Backward(tensorInput, tensorFlow, cuda_flag):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
        if cuda_flag:
            Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
        else:
            Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1)
    # end

    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
# end

class SpyNet(torch.nn.Module):
    def __init__(self, cuda_flag):
        super(SpyNet, self).__init__()
        self.cuda_flag = cuda_flag

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()

                self.moduleBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

            # end

            def forward(self, tensorInput):
                return self.moduleBasic(tensorInput)

        self.moduleBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(4)])

        self.load_state_dict(torch.load(SpyNet_model_dir + '/network-' + arguments_strModel + '.pytorch'), strict=False)


    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = [tensorFirst]
        tensorSecond = [tensorSecond]

        for intLevel in range(3):
            if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
                tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2))
                tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2))

        tensorFlow = tensorFirst[0].new_zeros(tensorFirst[0].size(0), 2,
                                              int(math.floor(tensorFirst[0].size(2) / 2.0)),
                                              int(math.floor(tensorFirst[0].size(3) / 2.0)))

        for intLevel in range(len(tensorFirst)):
            tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            # if the sizes of upsampling and downsampling are not the same, apply zero-padding.
            if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[0, 0, 0, 1], mode='replicate')
            if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[0, 1, 0, 0], mode='replicate')

            # input ：[first picture of corresponding level,
            # 		   the output of w with input second picture of corresponding level and upsampling flow,
            # 		   upsampling flow]
            # then we obtain the final flow. 最终再加起来得到intLevel的flow
            tensorFlow = self.moduleBasic[intLevel](torch.cat([tensorFirst[intLevel],
                                                               Backward(tensorInput=tensorSecond[intLevel],
                                                                        tensorFlow=tensorUpsampled,
                                                                        cuda_flag=self.cuda_flag),
                                                               tensorUpsampled], 1)) + tensorUpsampled
        return tensorFlow


class warp(torch.nn.Module):
    def __init__(self, h, w, cuda_flag):
        super(warp, self).__init__()
        self.height = h
        self.width = w
        if cuda_flag:
            self.addterm = self.init_addterm().cuda()
        else:
            self.addterm = self.init_addterm()

    def init_addterm(self):
        n = torch.FloatTensor(list(range(self.width)))
        horizontal_term = n.expand((1, 1, self.height, self.width))  # 第一个1是batch size
        n = torch.FloatTensor(list(range(self.height)))
        vertical_term = n.expand((1, 1, self.width, self.height)).permute(0, 1, 3, 2)
        addterm = torch.cat((horizontal_term, vertical_term), dim=1)
        return addterm

    def forward(self, frame, flow):
        """
        :param frame: frame.shape (batch_size=1, n_channels=3, width=256, height=448)
        :param flow: flow.shape (batch_size=1, n_channels=2, width=256, height=448)
        :return: reference_frame: warped frame
        """
        if True:
            flow = flow + self.addterm
        else:
            self.addterm = self.init_addterm()
            flow = flow + self.addterm

        horizontal_flow = flow[0, 0, :, :].expand(1, 1, self.height, self.width)  # 第一个0是batch size
        vertical_flow = flow[0, 1, :, :].expand(1, 1, self.height, self.width)

        horizontal_flow = horizontal_flow * 2 / (self.width - 1) - 1
        vertical_flow = vertical_flow * 2 / (self.height - 1) - 1
        flow = torch.cat((horizontal_flow, vertical_flow), dim=1)
        flow = flow.permute(0, 2, 3, 1)
        reference_frame = torch.nn.functional.grid_sample(frame, flow)
        return reference_frame


class ResNet(torch.nn.Module):
    """
    Three-layers ResNet/ResBlock
    reference: https://blog.csdn.net/chenyuping333/article/details/82344334
    """
    def __init__(self, task):
        super(ResNet, self).__init__()
        self.task = task
        self.conv_3x2_64_9x9 = torch.nn.Conv2d(in_channels=3 * 2, out_channels=64, kernel_size=9, padding=8 // 2)
        self.conv_3x7_64_9x9 = torch.nn.Conv2d(in_channels=3 * 7, out_channels=64, kernel_size=9, padding=8 // 2)
        self.conv_64_64_9x9 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, padding=8 // 2)
        self.conv_64_64_1x1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv_64_3_1x1 = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

    def ResBlock(self, x, aver):
        if self.task == 'interp':
            x = torch.nn.functional.relu(self.conv_3x2_64_9x9(x))
            x = torch.nn.functional.relu(self.conv_64_64_1x1(x))
        elif self.task in ['denoise', 'denoising']:
            x = torch.nn.functional.relu(self.conv_3x7_64_9x9(x))
            x = torch.nn.functional.relu(self.conv_64_64_1x1(x))
        elif self.task in ['sr', 'super-resolution']:
            x = torch.nn.functional.relu(self.conv_3x7_64_9x9(x))
            x = torch.nn.functional.relu(self.conv_64_64_9x9(x))
            x = torch.nn.functional.relu(self.conv_64_64_1x1(x))
        else:
            raise NameError('Only support: [interp, denoise/denoising, sr/super-resolution]')
        x = self.conv_64_3_1x1(x) + aver
        return x

    def forward(self, frames):
        aver = frames.mean(dim=1)
        x = frames[:, 0, :, :, :]
        for i in range(1, frames.size(1)):
            x = torch.cat((x, frames[:, i, :, :, :]), dim=1)
        result = self.ResBlock(x, aver)
        return result


class TOFlow(torch.nn.Module):
    def __init__(self, h, w, task, cuda_flag):
        super(TOFlow, self).__init__()
        self.height = h
        self.width = w
        self.task = task
        self.cuda_flag = cuda_flag

        self.SpyNet = SpyNet(cuda_flag=self.cuda_flag)  # SpyNet层
        # for param in self.SpyNet.parameters():  # fix
        #     param.requires_grad = False

        self.warp = warp(self.height, self.width, cuda_flag=self.cuda_flag)

        self.ResNet = ResNet(task=self.task)

    # frames should be TensorFloat
    def forward(self, frames, flows):
        """
        :param frames: [batch_size=1, img_num, n_channels=3, h, w]
        :return:
        """
        s1 = time.time()
        for i in range(frames.size(1)):
            frames[:, i, :, :, :] = normalize(frames[:, i, :, :, :])
        e1 = time.time()
        s2 = time.time()
        if self.cuda_flag:
            opticalflows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4)).cuda()
            warpframes = torch.empty(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4)).cuda()
        else:
            opticalflows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4))
            warpframes = torch.empty(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4))

        e2 = time.time()
        s3 = time.time()

        e3 = time.time()
        opticalflows[:, 0, 0, :, :] = 0
        opticalflows[:, 0, 1, :, :] = -flows[:, 0, :, :, :]/2
#        opticalflows[:, 0, 1, :, :] = flows[:, 0, :, :, :]
        opticalflows[:, 1, 0, :, :] = 0
        opticalflows[:, 1, 1, :, :] = -flows[:, 1, :, :, :]/2
#        opticalflows[:, 1, 1, :, :] = flows[:, 1, :, :, :]


# minus for zero direction

#        for i in process_index:
#            warpframes[:, i, :, :, :] = self.warp(frames[:, i, :, :, :], opticalflows[:, i, :, :, :])
        s4 = time.time()
        warpframes[:, 0, :, :, :] = self.warp(frames[:, 0, :, :, :], opticalflows[:, 0, :, :, :])
        warpframes[:, 1, :, :, :] = self.warp(frames[:, 1, :, :, :], opticalflows[:, 1, :, :, :])
        e4 = time.time()
        # warpframes: [batch_size=1, img_num=7, n_channels=3, height=256, width=448]

        s5 = time.time()
        Img = self.ResNet(warpframes)
        e5 = time.time()
        # Img: [batch_size=1, n_channels=3, h, w]

        s6 = time.time()
        Img = denormalize(Img)
        e6 = time.time()

        print("check point 1 : " + str(e1 - s1) + "s")
        print("check point 2 : " + str(e2 - s2) + "s")
        print("check point 3 : " + str(e3 - s3) + "s")
        print("check point 4 : " + str(e4 - s4) + "s")
        print("check point 5 : " + str(e5 - s5) + "s")
        print("check point 6 : " + str(e6 - s6) + "s")

        return Img





# %%

#run.py

frameFirstName = OutViewF
frameSecondName = OutViewB
flowFirstName = OutFlowF
flowSecondName = OutFlowB

model_name = 'interp'                                                       # select model
workplace = '.'

#frameFirstName = None
#frameSecondName = None
frameOutName = os.path.join(workplace, 'out.png')
#gpuID = None

gpuID = 0

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:

    if strOption == '--o':         # out frame
        frameOutName = strArgument
    elif strOption == '--gpuID':
        gpuID = int(strArgument)  



if gpuID == None:
    CUDA = False
else:
    CUDA = True
# ------------------------------
# 数据集中的图片长宽都弄成32的倍数了所以这里可以不用这个函数
# 暂时只用于处理batch_size = 1的triple
def Estimate(net, tensorFirst=None, tensorSecond=None, Firstfilename='', Secondfilename='', Firstflowname='', Secondflowname='', cuda_flag=False):
    """
    :param tensorFirst: 弄成FloatTensor格式的frameFirst
    :param tensorSecond: 弄成FloatTensor格式的frameSecond
    :return:
    """
    tensorFirst = torch.FloatTensor(Firstfilename.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
    tensorSecond = torch.FloatTensor(Secondfilename.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))

    flowFirst = Firstflowname
#    flowFirst  = mat_file['save_outflowF']
    flowFirst  = torch.from_numpy(flowFirst.astype(np.float32))
 
    flowSecond = Secondflowname
#    flowSecond = mat_file['save_outflowB']
    flowSecond = torch.from_numpy(flowSecond.astype(np.float32))

    tensorOutput = torch.FloatTensor()

    # check whether the two frames have the same shape
    assert (tensorFirst.size(1) == tensorSecond.size(1))
    assert (tensorFirst.size(2) == tensorSecond.size(2))

    intWidth = tensorFirst.size(2)
    intHeight = tensorFirst.size(1)

    # assert(intWidth == 448) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 256) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    if cuda_flag == True:
        tensorFirst = tensorFirst.cuda()
        tensorSecond = tensorSecond.cuda()
        tensorOutput = tensorOutput.cuda()
        flowFirst  = flowFirst.cuda()
        flowSecond = flowSecond.cuda()
    # end
    s1 = time.time()

    if True:
        s3 = time.time()
        tensorPreprocessedFirst = tensorFirst.view(1, 3, intHeight, intWidth)
        tensorPreprocessedSecond = tensorSecond.view(1, 3, intHeight, intWidth)
        flowPreprocessedFirst = flowFirst.view(1, 1, intHeight, intWidth)
        flowPreprocessedSecond = flowSecond.view(1, 1, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))  # 宽度弄成32的倍数，便于上下采样
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))  # 长度弄成32的倍数，便于上下采样

        tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(
            intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(
            intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        flowPreprocessedFirst = torch.nn.functional.interpolate(input=flowPreprocessedFirst, size=(
            intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        flowPreprocessedSecond = torch.nn.functional.interpolate(input=flowPreprocessedSecond, size=(
            intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        s4 = time.time()

        tensorFlow = torch.nn.functional.interpolate(
            input=net(torch.stack([tensorPreprocessedFirst, tensorPreprocessedSecond], dim=1), torch.stack([flowPreprocessedFirst, flowPreprocessedSecond], dim=1)), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tensorOutput.resize_(3, intHeight, intWidth).copy_(tensorFlow[0, :, :, :])
        tensorOutput = tensorOutput.permute(1, 2, 0)
    # end
    s2 = time.time()
    if True:
        tensorFirst = tensorFirst.cpu()
        tensorSecond = tensorSecond.cpu()
        tensorOutput = tensorOutput.cpu()
    # end
    print("total time : " + str(s2 - s1) + "s")
    print("temp  time : " + str(s4 - s3) + "s")

    return tensorOutput.detach().numpy()

# ------------------------------
if __name__ == '__main__':
    if CUDA:
        torch.cuda.set_device(gpuID)
    temp_img = frameFirstName
    height = temp_img.shape[0]
    width = temp_img.shape[1]

    intPreprocessedWidth = int(math.floor(math.ceil(width / 32.0) * 32.0))  # 宽度弄成32的倍数，便于上下采样
    intPreprocessedHeight = int(math.floor(math.ceil(height / 32.0) * 32.0))  # 长度弄成32的倍数，便于上下采样

    print('Loading TOFlow Net... ', end='')
    net = TOFlow(intPreprocessedHeight, intPreprocessedWidth, task='interp', cuda_flag=CUDA)
    net.load_state_dict(torch.load(os.path.join(workplace, 'toflow_models', model_name + '.pkl')))
    if CUDA:
        net.eval().cuda()
    else:
        net.eval()

    print('Done.')

    # ------------------------------
    # generate(net=net, model_name=model_name, f1name=os.path.join(test_pic_dir, 'im1.png'),
    #         f2name=os.path.join(test_pic_dir, 'im3.png'), fname=outputname)
    print('Processing...')
    predict = Estimate(net, Firstfilename=frameFirstName, Secondfilename=frameSecondName, Firstflowname=flowFirstName, Secondflowname=flowSecondName, cuda_flag=CUDA)
    #print(predict, np.min(predict), np.max(predict))

    

    plt.imsave(frameOutName, predict)
    print('%s Saved.' % frameOutName)

# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:12:03 2019

@author: Ziyu Su
"""

import cv2
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
from os import path
from scipy.ndimage import measurements
from skimage.measure import regionprops
#%%
sqrt=lambda x:np.sqrt(x)

def rgb2lab(img):
    lms_m=np.array([[0.3811,0.1967,0.0241],[0.5783,0.7244,0.1288],[0.0402,0.0782,0.8444]])
    rgb2lms=img@lms_m
    #loglms=np.log(rgb2lms)
    lab_m=np.array([[1/sqrt(3),1/sqrt(6),1/sqrt(2)],[1/sqrt(3),1/sqrt(6),-1/sqrt(2)],[1/sqrt(3),-2/sqrt(6),0]])
    lms2lab=rgb2lms@lab_m
    return lms2lab

def lab2rgb(img):
    lms_m=np.array([[sqrt(3)/3,sqrt(3)/3,sqrt(3)/3],[sqrt(6)/6,sqrt(6)/6,-sqrt(6)/3],[sqrt(2)/2,-sqrt(2)/2,0]])
    lab2lms=img@lms_m
    
    #powlms=np.exp(lab2lms)
    rgb_m=np.array([[4.4679,-1.2186,0.0497],[-3.5873,2.3809,-0.2439],[0.1193,-0.1624,1.2045]])
    lms2rgb=lab2lms@rgb_m
    return lms2rgb

def areasort(I,idx,img):
    props=regionprops(I)
    area=[]
    attr=['area','label']
    for i in range(idx):
        area+=[[]]
        for j in range(len(attr)):
            area[i]+=[getattr(props[i],attr[j])]
    area=np.array(area)
    area=area[area[:,0].argsort()]
    for i in range(idx-1):
        img[I==area[i,1]]=0
#    img[I==j] for j in
    result=img
    return result
def block_remove(im):
    ILabel, nFeatures = measurements.label(im)
    a=areasort(ILabel,nFeatures,im)
    a=255*(a-255)#take inverse map
    
    ILabel1, nFeatures1 = measurements.label(a)
    b=(areasort(ILabel1,nFeatures1,a)-255)*255
    plt.imshow(b)
    plt.colorbar()
    return b
#%%
#dirname=path.join(os.getcwd(),'images/real')
#
#img=cv2.imread(dirname+'/'+'10.jpg')


#%%
num=1
while num<=1:
    print('>>>>>>>>num{}'.format(num))
    imgnumber=num
    stylename='style1.jpg'
    skyname='sky3.jpg'
    
    
    dirname=path.join(os.getcwd(),'demo_image')
    data_dirname=path.join(os.getcwd(),'output')
    imgname=dirname+'/'+str(imgnumber)+'.jpg'
    maskname=data_dirname+'/mask'+str(imgnumber)+'.jpg'
    colorname=data_dirname+'/trancolor'+str(imgnumber)+'.jpg'
    
    #equname=data_dirname+'/equ'+str(imgnumber)+'.jpg'
    mergename=data_dirname+'/a_output'+str(imgnumber)+'.jpg'
    img=cv2.imread(imgname)
    #%%
    '''
    sky segmentation
    '''
    
    low=np.array([[0,0,216],[95,0,204],[100,0,153],[105,0,88]])
    high=np.array([[180,13,255],[125,25,255],[115,128,255],[110,255,255]])
    img_s=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    #hsv
    h,s,v=cv2.split(img_s)
    v=cv2.equalizeHist(v)
    hsv=cv2.merge((h,s,v))
    imgThr=np.zeros(np.shape(hsv)[0:2])
    
    for i in range(4):
        temp=cv2.inRange(hsv,low[i],high[i])
        imgThr=imgThr+temp
    imgThr=np.uint8(255*(imgThr>0))# opencv threshold
    
    imgThr=cv2.medianBlur(imgThr,9)
    
    kernel=np.ones((5,5),np.uint8)
    imgThr=cv2.morphologyEx(imgThr,cv2.MORPH_OPEN,kernel,iterations=10)
    imgThr=cv2.medianBlur(imgThr,9)
    
    
    true_Thr=block_remove(imgThr)
    
    
    #newname=os.getcwd()+'/data'+'/'+'mask3.jpg'
    cv2.imwrite(maskname,true_Thr)
    
    mask=np.copy(true_Thr)
    
    
    #%%#################################
    
    '''
    edge mask create
    '''
    mask_01=1*(mask==0)
    #img=cv2.imread('7.jpg')
    r,g,b=cv2.split(img)
    edge1=np.uint32(cv2.Canny(np.uint8(r*mask_01),50,210))
    edge2=np.uint32(cv2.Canny(np.uint8(g*mask_01),50,210))
    edge3=np.uint32(cv2.Canny(np.uint8(b*mask_01),50,210))
    edge=edge1+edge2+edge3-3*np.uint32(cv2.Canny(np.uint8(mask),50,210))
 

    edge[edge>800]=0
    edge[edge>200]=1
    
    edge=np.uint8(edge)
    mask3d=cv2.merge((edge,edge,edge))
    
    #%%
    '''
    bilateral filtering
    smooth, reduce detail
    '''
    img_smooth=cv2.bilateralFilter(img, 10, 50, 50)
    #cv2.imwrite("process3.jpg",img_smooth)
    #%%
    #sharpen=np.copy(img2)
    #sharpen[mask3d==1]=10
    ##cv2.imwrite("sharpen.jpg",sharpen)
    ##%%
    #cv2.imwrite("process7.jpg",sharpen)
    #%%###################################################
    '''
    color modulation
    '''
    #-----HSV------#
    style=cv2.imread(stylename)
    
    lab_sty=cv2.cvtColor(style,cv2.COLOR_BGR2HSV)
    
    lab_img=cv2.cvtColor(img_smooth,cv2.COLOR_BGR2HSV)
    l,a,b=np.float64(cv2.split(lab_img))
    _,aa,bb=np.float64(cv2.split(lab_sty))   
    
    #l1=((l-np.mean(l))*np.std(ll)/np.std(l))+np.mean(ll)
    a1=((a-np.mean(a))*np.std(aa)/np.std(a))+np.mean(aa)
    b1=((b-np.mean(b))*np.std(bb)/np.std(b))+np.mean(bb)
    
    l1=l
    new=cv2.merge((l1,a1.clip(0,255),b1.clip(0,255)))
    
    #new1=lab2rgb(new)
    new1=np.uint8(np.around(new))
    
    img_color=cv2.cvtColor(new1,cv2.COLOR_HSV2BGR)
    
    cv2.imwrite(colorname,img_color)
    #%%
    ##-----LAB-------#
    #style=cv2.imread('style3.jpg')
    #
    #lab_sty=rgb2lab(style)
    #lab_img=rgb2lab(img2)
    #l,a,b=np.float64(cv2.split(lab_img))
    #ll,aa,bb=np.float64(cv2.split(lab_sty))   
    #
    #l1=((l-np.mean(l))*np.std(ll)/np.std(l))+np.mean(ll)
    #a1=((a-np.mean(a))*np.std(aa)/np.std(a))+np.mean(aa)
    #b1=((b-np.mean(b))*np.std(bb)/np.std(b))+np.mean(bb)
    #
    #new=cv2.merge((l1,a1,b1))
    #new=lab2rgb(new)
    #new=new.clip(0,255)
    #new1=np.uint8(np.around(new))
    #cv2.imwrite('trancolor.jpg',new1)
    
    #%%####################################################
    
    '''
    sky merge
    mask,sky,imgOri
    '''
    
    sky=cv2.imread(skyname)
    img_color=cv2.imread(colorname)
    
    mask=cv2.imread(maskname,0)
    
    contours=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt=contours[0]
    
    x,y,w,h=cv2.boundingRect(cnt)
    print(x,y,w,h)
    if w==0 or h==0:
        output=img_color
        
    img_x=len(img_color[0])
    img_y=len(img_color[1])
    sky_x=len(sky[0])
    sky_y=len(sky[1])
    scale_x=w*1.0/sky_x
    
    skyscale=cv2.resize(sky,(img_color.shape[1],img_color.shape[0]),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("scale_sky.jpg",skyscale)
    center=[int((x+w)/2),int((y+h)/2)]
    center=(center[0],center[1])
    print(center)
    
    img_merge=cv2.seamlessClone(skyscale,img_color,mask,center,cv2.NORMAL_CLONE)
    
    #cv2.imwrite("merge7.jpg",output)
    
    
    #cv2.imwrite("merge3.jpg",output)
    #%%
    '''
    edge superimpose
    '''
    sharpen=np.copy(img_merge)
    sharpen[mask3d==1]=100
    #cv2.imwrite("sharpen.jpg",sharpen) 
#    r,g,b=cv2.split(sharpen)
#    r=cv2.equalizeHist(r)
#    g=cv2.equalizeHist(g)
#    b=cv2.equalizeHist(b)
#    equ=cv2.merge((r,g,b))
#    cv2.imwrite(equname,equ)
    cv2.imwrite(mergename,sharpen)
    
    
    num+=1


#%%
#cv2.namedWindow("image",cv2.WINDOW_NORMAL)
#cv2.imshow("image",skyapt)
#cv2.waitKey(0)

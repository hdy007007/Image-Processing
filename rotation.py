# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:48:05 2017

@author: ICT
"""

import cv2 as cv
import math
import matplotlib.pyplot as plt
import numpy as np


def corRotationTrans(p,rad):#旋转坐标转换
    p1 = [0,0]
    p1[0] = p[0]*math.cos(rad) - p[1]*math.sin(rad) #y
    p1[1] = p[1]*math.cos(rad) + p[0]*math.sin(rad) #x
    return p1

def corInvRotationTrans(p,rad):#逆旋转坐标转换
    p1 = [0,0]
    p1[0] = p[1]*math.sin(rad) + p[0]*math.cos(rad)
    p1[1] = p[1]*math.cos(rad) - p[0]*math.sin(rad)
    return p1

def corTransToCenter(p,imgSize): #转换成中心为原点的坐标系
    p1 = [0,0]
    center = [imgSize[0]/2,imgSize[1]/2]
    p1[0] = center[0] - p[0]
    p1[1] = p[1] - center[1]
    return p1

def corTransRestore(p,imgSize): #转换回左上角为原点的坐标系
    p1 = [0,0]
    center = [imgSize[0]/2,imgSize[1]/2]
    p1[1] = center[1] + p[1] #x
    p1[0] = center[0] - p[0] #y
    return p1

def corAfterRot(p,rad,imgSize1,imgSize2):#旋转后的新坐标
    p1 = corTransToCenter(p,imgSize1)
    p2 = corRotationTrans(p1,rad)
    p3 = corTransRestore(p2,imgSize2)
    return p3

def corAfterInvRot(p,rad,imgSize1,imgSize2):#逆旋转后的新坐标
    p1 = corTransToCenter(p,imgSize1)
    p2 = corInvRotationTrans(p1,rad)
    p3 = corTransRestore(p2,imgSize2)
    return p3

def getNewSize(rad,size):#旋转后新图像的大小
    p1 = corRotationTrans(corTransToCenter((0,0),size),rad)
    p2 = corRotationTrans(corTransToCenter((0,size[1]),size),rad)
    p3 = corRotationTrans(corTransToCenter((size[0],0),size),rad)
    p4 = corRotationTrans(corTransToCenter((size[0],size[1]),size),rad)
    newHeight = int(max(p1[0],p2[0],p3[0],p4[0]) - min(p1[0],p2[0],p3[0],p4[0])) + 1
    newWidth = int(max(p1[1],p2[1],p3[1],p4[1]) - min(p1[1],p2[1],p3[1],p4[1])) + 1
    return [newHeight,newWidth]


def getRotatedImg(img,angle):#双线性插值，角落部分用原图片的边界所有像素均值进行填充
    rad = math.radians(angle)
    height,width = img.shape[:2]
    [newHeight,newWidth] = getNewSize(rad,[height,width])
    fourSidesPix = np.zeros([2 * height + 2 * width - 4,3],dtype = np.uint8)
    fourSidesCor = np.zeros([2 * height + 2 * width - 4,2],dtype = np.uint16)
    fourSidesCorAfterRot = np.zeros([2 * height + 2 * width - 4,2],dtype = np.uint16)
    imageRegion = np.zeros([newHeight,2],dtype = np.uint8)
    sidePointNum = 0;
    #fourSides[0:height,:] = img[:,0,:]
    #fourSides[height:height + width - 1,:] = img[0,1:width,:]
    #fourSides[height + width - 1:height + width + height - 2,:] = img[1:height,width - 1,:]
    #fourSides[height + width + height - 2:height + width + height + width- 3] = img[height - 1,1:width - 1,:]
    
    for i in range(0,height):
        for j in range(0,width):
            if i == 0 or j == 0 or i == height - 1 or j == width - 1:
                fourSidesPix[sidePointNum,:] = img[i,j,:]
                fourSidesCor[sidePointNum,:] = [i,j]
                sidePointNum = sidePointNum + 1
    
    index = 0
    for i in fourSidesCor:
        fourSidesCorAfterRot[index,:] = corAfterRot(i,rad,(height,width),(newHeight,newWidth))
        index = index + 1
    fourSidesCorAfterRotSorted = fourSidesCorAfterRot[fourSidesCorAfterRot[:,0].argsort(),:] 
    
    imageRegion[0,:] = fourSidesCorAfterRotSorted[0,1]
    minNum = fourSidesCorAfterRotSorted[0,1]
    maxNum = fourSidesCorAfterRotSorted[0,1]
    index = 0
    for i in range(0,len(fourSidesCorAfterRotSorted)):
        if minNum > fourSidesCorAfterRotSorted[i,1]:
            minNum = fourSidesCorAfterRotSorted[i,1]
        if maxNum < fourSidesCorAfterRotSorted[i,1]:
            maxNum = fourSidesCorAfterRotSorted[i,1]
    
        if i == len(fourSidesCorAfterRotSorted) - 1:
            imageRegion[index,0] = minNum
            imageRegion[index,1] = maxNum          
        elif  fourSidesCorAfterRotSorted[i + 1,0] != fourSidesCorAfterRotSorted[i,0] :
            imageRegion[index,0] = minNum
            imageRegion[index,1] = maxNum
            index += 1
            minNum = fourSidesCorAfterRotSorted[i + 1,1]
            maxNum = fourSidesCorAfterRotSorted[i + 1,1]
            
    #p = corTransToCenter((40,20),(40,20))
    #print p 
    newImg = np.zeros([newHeight,newWidth,3],dtype = np.uint8)
    newImg[:,:,0] = np.uint8(np.mean(fourSidesPix[:,0]))
    newImg[:,:,1] = np.uint8(np.mean(fourSidesPix[:,1]))
    newImg[:,:,2] = np.uint8(np.mean(fourSidesPix[:,2]))
    for i in range(0,newHeight):
        for j in range(imageRegion[i,0],imageRegion[i,1] + 1):
            p = corAfterInvRot((i,j),rad,(newHeight,newWidth),(height,width))
            p[0] = min(p[0],height - 1)
            p[0] = max(p[0],0)
            p[1] = min(p[1],width - 1)
            p[1] = max(p[1],0)

            u = p[0] - int(p[0])
            v = p[1] - int(p[1])
            
            if int(p[0]) == height - 1 or int(p[1]) == width - 1:
                newImg[i,j,:] = img[int(p[0]),int(p[1]),:]
            else:
                newImg[i,j,:] = (1-u)*(1-v)*img[int(p[0]),int(p[1]),:] + (1-u)*v*img[int(p[0]),int(p[1])+1,:] + u*(1-v)*img[int(p[0])+1,int(p[1]),:] + u*v*img[int(p[0])+1,int(p[1])+1,:]
    return newImg
        

def main():
    img = cv.imread(r'E:\\trafficImage\\cutoutTrain\\pl5_55362_0.jpg')
    height,width = img.shape[:2]
    angle = -60
    
    newImg= getRotatedImg(img,angle)
    cv.imwrite('1111.jpg',newImg)
            
    plt.imshow(newImg)
    
if __name__ == '__main__':
    main()
from PIL import Image, ImageOps
import numpy as np
import os
import cv2
import math as m
import os
import matplotlib.pyplot as plt
import random
from random import randrange
from os import path

max_masks = 100000
taken = {}

# the following is for homography
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
#     rows,cols = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    
    # Brightness 
    

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    img = augment_brightness_camera_images(img)
    
    return img

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


## Generates real damage masks from the damaged input image.
##Assuming damage is white, fetch all the pixels from range (0.85* max. intensity pixel - max. intenisty pixel).
def generateMasks(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    maxVal= np.max(img)
#     print("max value "+ str(np.max(img)))
    threshold = 0.85 * maxVal
    ret, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)
#     print("mask.shape " + str(mask.shape))
    
    width, height = mask.shape
    mask_bin = mask>200
    count = 0
    ind = [0 for i in range(width*height)]
    for i in range(width):
        for j in range(height):
            if(mask_bin[i][j] == True):
                ind[count] = (j)*width + i
                count = count + 1
    ##end of for
    sal = ind[0:count-1]
    np.random.shuffle(sal)

    mask_1 = mask_bin < 0
    for i in range(int(m.floor((count-1)/3))):
        val = sal[i]
        row = m.floor(val/width)
        col = m.floor(val - (row * width))
        mask_1[int(m.floor(col))][int(m.floor(row))] = True
    mask_1 = mask_1.astype(np.uint8)
    mask_2 = mask_bin <0
    for i in range(int(m.floor((count-1)/3+1)), int(m.floor(2*(count-1)/3))):
        val = sal[i]
        row = m.floor(val/width)
        col = m.floor(val - (row * width))
        mask_2[int(m.floor(col))][int(m.floor(row))] = True
    mask_2 = mask_2.astype(np.uint8)
    mask_3 = mask_bin <0
    for i in range(int(m.floor(2*(count-1)/3+1)), int(m.floor((count-1)/1))):
        val = sal[i]
        row = m.floor(val/width)
        col = m.floor(val - (row * width))
        mask_3[int(m.floor(col))][int(m.floor(row))] = True
    mask_3 = mask_3.astype(np.uint8)
    mask_4 = (mask_1 | mask_2 | mask_3)

    
    return mask_4


def rotateImages(img, dest):
    # for each image, save a flipped/rotated version
    r90 = cv2.rotate(img, cv2.ROTATE_90)
    plt.imsave(dest+'/mask{}.png'.format(getUniqueID()), (r90 * 255).astype(np.uint8), cmap='gray')

    r180 = cv2.rotate(img, cv2.ROTATE_180)
    plt.imsave(dest+'/mask{}.png'.format(getUniqueID()), (r180 * 255).astype(np.uint8), cmap='gray')

    r270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    plt.imsave(dest+'/mask{}.png'.format(getUniqueID()), (r270 * 255).astype(np.uint8), cmap='gray')

    rFlip = cv2.flip(img, 1)
    plt.imsave(dest+'/mask{}.png'.format(getUniqueID()), (rFlip * 255).astype(np.uint8), cmap='gray')
    
    return


def getUniqueID():
    num = 0
    while True:
        num = random.randint(1, max_masks)
        if num not in taken:
            taken[num] = 1
            break
    return num


def getHomography(img, dest):    
    for i in range(100):
        img = transform_image(image,20,10,5)
        if np.sum(img) != 0:
            # for save initial image
            plt.imsave(dest+'/mask{}.png'.format(getUniqueID()), (img * 255).astype(np.uint8), cmap='gray')

            # flip and rotate image by 90, 180, 270 degrees, and save
            rotateImages(img, dest)
    return


def getMasks(src, dest):
    for image in os.listdir(src):
        if (image.endswith(".png") or image.endswith(".jpg")):
            # generate mask
            test = generateMasks(src+'/'+image)

            # get random crops (say 10 samples)
            x,y = test.shape   
            matrix = 256
            sample = 10
            
            if x <= matrix:
                txt = "/mask{id}.png".format(id=getUniqueID())
                test.save(dest+txt, 'PNG')
                # apply augmentation to given image
                getHomography(test, dest)

            else:
                for i in range(sample):
                    x1 = randrange(0, x - matrix)
                    y1 = randrange(0, y - matrix)
                    # img = test.crop((x1, y1, x1 + matrix, y1 + matrix))
                    img = test[x1: x1+matrix, y1: y1+matrix]
                    # do not save image if just black square
                    if np.sum(img) == 0:
                        continue

                    # save mask
                    txt = "/mask{id}.png".format(id=getUniqueID())
                    img.save(dest+txt, 'PNG')

                    # apply augmentation to given image
                    getHomography(img, dest)
                    
    return

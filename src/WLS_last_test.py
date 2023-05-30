# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:31:04 2023

@author: Mathieu Chesneau
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as ssp
import time

start = time.time()

#---------------------------------------------------------------

# Return the luminance of Color c.

def luminance(img):
    R=img[:,:,0]
    G=img[:,:,1]
    B=img[:,:,2]
    return (0.299 * R) + (0.587 * G) + (0.114 * B)

#---------------------------------------------------------------

def extraction_image(nomFichier):
    img = Image.open(nomFichier)
    return np.array(img)

#---------------------------------------------------------------

# Definition of the forward difference operators

def deriveur_x(img,i,j):
    return abs((img[i+1,j])-img[i,j])

def deriveur_y(img,i,j):
    return abs((img[i,j+1])-img[i,j])

#---------------------------------------------------------------

# Definition of the parameters of the problem

epsilon=0.0001
alpha=1.6
lbda=10

def WLSFilter(epsilon,alpha,lbda,img):
    (row, col, RGB) = img.shape
    plt.figure("Image Ordered")
    plt.imshow(img)
    plt.title("INITAIAL Smoothed Image")
    img_t = img.transpose(1, 0, 2)
    plt.figure("Image OKDZ?rdered")
    plt.imshow(img_t)
    plt.title("INITAIAL Smoothed kjoImage")
    nbr_pix=col*row
    logY = np.log(luminance(img) + np.ones((row, col)))
            
    #---------------------------------------------------------------
    # Calculation of the ax coefficients
    #---------------------------------------------------------------

    ax = np.zeros((row,col))
    ay = np.zeros((row, col))

    for i in range(0,row-1):
        for j in range(0,col-1):
            ax[i,j]=((deriveur_x(logY,i,j)**alpha)+epsilon)**-1
            ay[i,j]=((deriveur_y(logY,i,j)**alpha)+epsilon)**-1
    ay = ay.transpose()
    ax_vec=ax.reshape(nbr_pix, 1)
    ay_vec=ay.reshape(nbr_pix, 1)
    img_vec=img.reshape(1, nbr_pix, 3)
    img_vec_t=img_t.reshape(1, nbr_pix, 3)

    #---------------------------------------------------------------
    # Generation of the Dx matrix
    #---------------------------------------------------------------

    main_diag=np.ones((1,nbr_pix))
    side_diag=-1*(np.ones((1,nbr_pix - 1)))
    diagonals=[main_diag,side_diag]
    base=ssp.diags(diagonals,[0,1],shape=(nbr_pix, nbr_pix))
    base=ssp.csr_matrix(base)

    AX=ssp.diags(ax_vec.transpose(),[0])
    AY=ssp.diags(ay_vec.transpose(),[0])
    AY=ssp.csr_matrix(AY)
    AX=ssp.csr_matrix(AX)

    Id=ssp.identity(AX.shape[1])

    #---------------------------------------------------------------
    # Generation of the Lg matrix
    #---------------------------------------------------------------

    Lgx=(ssp.csr_matrix.transpose(base)@AX@base)

    #---------------------------------------------------------------
    # Reconstruction of the image
    #---------------------------------------------------------------

    H1= Id + lbda*Lgx
    # H2 = lbda * Lgy
    img_r = np.transpose(img_vec[:,:,0])
    img_g = np.transpose(img_vec[:,:,1])
    img_b = np.transpose(img_vec[:,:,2])
    img_r_t = np.transpose(img_vec_t[:,:,0])
    img_g_t = np.transpose(img_vec_t[:,:,1])
    img_b_t = np.transpose(img_vec_t[:,:,2])

    New_Img_R=ssp.linalg.spsolve(H1,img_r)
    New_Img_G=ssp.linalg.spsolve(H1,img_g)
    New_Img_B=ssp.linalg.spsolve(H1,img_b)

    New_Img=np.zeros([1,nbr_pix,3])
    New_Img[:,:,0]=(1/255)*New_Img_R
    New_Img[:,:,1]=(1/255)*New_Img_G
    New_Img[:,:,2]=(1/255)*New_Img_B
    New_Img_t=np.zeros([1,nbr_pix,3])

    New_Img=New_Img.reshape([row,col,3])
    print(New_Img.shape)
    print(New_Img_t.shape)
    New_Img=New_Img
    return New_Img

#---------------------------------------------------------------

img = extraction_image("../data/fala.jpg")
img_t = img.transpose(1, 0, 2)
# img=np.array([[[5,5,3],[4,5,6],[25,25,255]],
#               [[2,5,3],[4,5,6],[5,55,25]],
#               [[71,4,3],[4,54,6],[255,25,255]],
#               [[15,12,3],[47,5,6],[25,55,55]],
#               [[48,16,3],[41,5,6],[25,25,5]]])


New_Img=WLSFilter(epsilon, alpha, lbda, img)
New_Img_t=WLSFilter(epsilon, alpha, lbda, img_t)



# print(ssp.csr_matrix.toarray(DX))
# print("\n")
# print(ssp.csr_matrix.toarray(DY))
# print("\n")


plt.figure("Image 1")
plt.imshow(New_Img)
plt.title("Smoothed Image")

plt.figure("Image 4")
plt.imshow(New_Img_t.transpose(1, 0, 2))
plt.title("Smoothed Image")

plt.figure("Image 5")
plt.imshow(0.5 * New_Img_t.transpose(1, 0, 2) + 0.5 * New_Img)
plt.title("Smoothed Image")

plt.figure("Image 2")
plt.imshow(img)
plt.title("Original Image")
plt.show()

end = time.time()
print(end - start)
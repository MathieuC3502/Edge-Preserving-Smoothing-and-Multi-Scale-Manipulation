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
lbda=3

def WLSFilter(epsilon,alpha,lbda,img):
    (row, col, RGB) = img.shape
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
    # ay = ay.transpose()
    ax_vec=ax.reshape(nbr_pix, 1)
    ay_vec=ay.reshape(nbr_pix, 1)
    img_vec=img.reshape(1, nbr_pix, 3)

    #---------------------------------------------------------------
    # Generation of the Dx matrix
    #---------------------------------------------------------------

    main_diag=np.ones((1,nbr_pix))
    side_diag=-1*(np.ones((1,nbr_pix - 1)))
    diagonals=[main_diag,side_diag]
    base=ssp.diags(diagonals,[0,1],shape=(nbr_pix, nbr_pix))
    base=ssp.csr_matrix(base)

    # basey = ssp.diags(diagonals,[0,row],shape=(nbr_pix, nbr_pix))
    # basey = ssp.csr_matrix(basey)
    # main_diag=np.ones((1,col-1))
    # side_diag=-1*(np.ones((1,col-1)))
    # diagonals=np.array([main_diag,side_diag])
    # base=ssp.diags(diagonals,[0,1],shape=(col-1,col))

    # DX=base
    # DX=ssp.csr_matrix(DX)
    # DX._shape = (DX.shape[0]+1,DX.shape[1])
    # DX.indptr = np.hstack((DX.indptr,DX.indptr[-1]))

    # for i in range(row-1):
    #     DX=ssp.bmat([[DX,None],[None,base]])
    #     DX=ssp.csr_matrix(DX)
    #     DX._shape = (DX.shape[0]+1,DX.shape[1])
    #     DX.indptr = np.hstack((DX.indptr,DX.indptr[-1]))

    # print("Taille DX: " + str(DX.shape))

    #---------------------------------------------------------------
    # Generation of the Dy matrix
    #---------------------------------------------------------------

    # main_diagy=np.ones((1,col-1))
    # side_diagy=-1*(np.ones((1,col-1)))
    # diagonalsy=np.array([main_diagy,side_diagy])
    # basey=ssp.diags(diagonalsy,[0,1],shape=(nbr_pix, nbr_pix))

    # DY=basey

    #A way to add a row of zeros at the bottom of a sparse matrix
    #Source : https://stackoverflow.com/questions/4695337/expanding-adding-a-row-or-column-a-scipy-sparse-matrix
    # DY=ssp.csr_matrix(DY)
    # DY._shape = (DY.shape[0]+1,DY.shape[1])
    # DY.indptr = np.hstack((DY.indptr,DY.indptr[-1]))

    # for i in range(col-1):
    #     DY=ssp.bmat([[DY,None],[None,basey]])
    #     DY=ssp.csr_matrix(DY)
    #     DY._shape = (DY.shape[0]+1,DY.shape[1])
    #     DY.indptr = np.hstack((DY.indptr,DY.indptr[-1]))
    # print("Taille DY: " + str(DY.shape))
    # #---------------------------------------------------------------
    # # Generation of the AX and AY matrixes
    # #---------------------------------------------------------------

    AX=ssp.diags(ax_vec.transpose(),[0])
    AY=ssp.diags(ay_vec.transpose(),[0])
    AY=ssp.csr_matrix(AY)
    AX=ssp.csr_matrix(AX)

    Id=ssp.identity(AX.shape[1])

    #---------------------------------------------------------------
    # Generation of the Lg matrixs
    #---------------------------------------------------------------

    Lg=(ssp.csr_matrix.transpose(base)@AX@base) #+ (ssp.csr_matrix.transpose(basey)@AY@basey)
    # Lg=(ssp.csr_matrix.transpose(DX)@AX@DX) + (ssp.csr_matrix.transpose(DY)@AY@DY)

    #---------------------------------------------------------------
    # Reconstruction of the image
    #---------------------------------------------------------------

    H=Id +(lbda*Lg)
    img_r = np.transpose(img_vec[:,:,0])
    img_g = np.transpose(img_vec[:,:,1])
    img_b = np.transpose(img_vec[:,:,2])
    New_Img_R=ssp.linalg.spsolve(H,img_r)
    New_Img_G=ssp.linalg.spsolve(H,img_g)
    New_Img_B=ssp.linalg.spsolve(H,img_b)
    New_Img=np.zeros([1,nbr_pix,3])
    New_Img[:,:,0]=New_Img_R
    New_Img[:,:,1]=New_Img_G
    New_Img[:,:,2]=New_Img_B

    New_Img=New_Img.reshape([row,col,3])
    # MatX = ssp.csr_matrix.toarray(DX)
    # Maty = ssp.csr_matrix.toarray(DY)
    # MatAX = ssp.csr_matrix.toarray(AX)
    # MatAY = ssp.csr_matrix.toarray(AY)
    # MatLg = ssp.csr_matrix.toarray(Lg)
    # r = Lg@Id@ax@ay@Lg
    diff = img - New_Img
    print(img.shape)
    print(New_Img.shape)
    return New_Img / 255 ,ax,ay, diff / 255

#---------------------------------------------------------------

img = extraction_image("../data/fala.jpg")
# img=np.array([[[5,5,3],[4,5,6],[25,25,255]],
#               [[2,5,3],[4,5,6],[5,55,25]],
#               [[71,4,3],[4,54,6],[255,25,255]],
#               [[15,12,3],[47,5,6],[25,55,55]],
#               [[48,16,3],[41,5,6],[25,25,5]]])


New_Img,ax,ay, diff=WLSFilter(epsilon, alpha, lbda, img)

# print(ssp.csr_matrix.toarray(DX))
# print("\n")
# print(ssp.csr_matrix.toarray(DY))
# print("\n")

A = np.zeros((6,1))
B = np.zeros((5,2))
plt.figure("Image 1")
plt.imshow(New_Img)
plt.title("Smoothed Image")

plt.figure("Image 2")
plt.imshow(img)
plt.title("Original Image")

plt.figure("Details")
plt.imshow(diff)
plt.title("Original Image")

plt.figure("Details and image ")
plt.imshow(5 * diff + New_Img)
plt.title("Original Image")
plt.show()

end = time.time()
print(end - start)


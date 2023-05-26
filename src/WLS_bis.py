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
    return (.299 * R) + (.587 * G) + (.114 * B)

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

def deriveur_y_bis(img,i,j):
    return abs((2*img[i,j])-img[i,j-1]-img[i,j+1])

#---------------------------------------------------------------

# Definition of the parameters of the problem

epsilon=0.0001
alpha=1.6
lbda=3

def WLSFilter(epsilon,alpha,lbda,img):
    col = img.shape[1]
    row = img.shape[0]
    nbr_pix=col*row

    Y=np.array(luminance(img))      # Luminance plane of the image
    l=np.zeros(Y.shape)             # Log-Luminance plane of the image

    for i in range(row):
        for j in range(col):
            l[i,j]=np.log(Y[i,j]+1)   # Log-Luminance plane of the image
            
            
    #---------------------------------------------------------------
    # Calculation of the ax coefficients
    #---------------------------------------------------------------

    ax = np.zeros((row,col))
    ay = np.zeros((row, col))

    for i in range(0,row-1):
        for j in range(0,col-1):
            ax[i,j]=((deriveur_x(l,i,j)**alpha)+epsilon)**-1
            # ay[i,j]=((deriveur_y(l,i,j)**alpha)+epsilon)**-1
    img_t = img.transpose()
    col_t = img_t.shape[1]
    row_t = img_t.shape[0]

    for i in range(0,row_t-1):
        for j in range(0,col_t-1):
            # ax[i,j]=((deriveur_x(l,i,j)**alpha)+epsilon)**-1
            ay[i,j]=((deriveur_x(l,i,j)**alpha)+epsilon)**-1
    ax_vec=ax.reshape(nbr_pix, 1)
    ay_vec=ay.reshape(nbr_pix, 1)
    img_vec=img.reshape(1, nbr_pix, 3)

    #---------------------------------------------------------------
    # Generation of the Dx matrix
    #---------------------------------------------------------------

    main_diag=np.ones((1,col-1))
    side_diag=-1*(np.ones((1,col-1)))
    diagonals=np.array([main_diag,side_diag])
    base=ssp.diags(diagonals,[0,1],shape=(col-1,col))

    DX=base
    DX=ssp.csr_matrix(DX)
    DX._shape = (DX.shape[0]+1,DX.shape[1])
    DX.indptr = np.hstack((DX.indptr,DX.indptr[-1]))

    for i in range(row-1):
        DX=ssp.bmat([[DX,None],[None,base]])
        DX=ssp.csr_matrix(DX)
        DX._shape = (DX.shape[0]+1,DX.shape[1])
        DX.indptr = np.hstack((DX.indptr,DX.indptr[-1]))

    print("Taille DX: " + str(DX.shape))

    #---------------------------------------------------------------
    # Generation of the Dy matrix
    #---------------------------------------------------------------

    main_diag=np.ones([1,row-1])
    side_diag=-1*(np.ones([1,row-1]))
    diagonals=np.array([main_diag,side_diag])
    base=ssp.diags(diagonals,[0,1],shape=(row-1,row))

    DY=base

    #A way to add a row of zeros at the bottom of a sparse matrix
    #Source : https://stackoverflow.com/questions/4695337/expanding-adding-a-row-or-column-a-scipy-sparse-matrix
    DY=ssp.csr_matrix(DY)
    DY._shape = (DY.shape[0]+1,DY.shape[1])
    DY.indptr = np.hstack((DY.indptr,DY.indptr[-1]))

    for i in range(col-1):
        DY=ssp.bmat([[DY,None],[None,base]])
        DY=ssp.csr_matrix(DY)
        DY._shape = (DY.shape[0]+1,DY.shape[1])
        DY.indptr = np.hstack((DY.indptr,DY.indptr[-1]))
    print("Taille DY: " + str(DY.shape))
    #---------------------------------------------------------------
    # Generation of the AX and AY matrixes
    #---------------------------------------------------------------

    AX=ssp.diags(ax_vec.transpose(),[0])
    AY=ssp.diags(ay_vec.transpose(),[0])

    Id=ssp.identity(AX.shape[1])

    #---------------------------------------------------------------
    # Generation of the Lg matrix
    #---------------------------------------------------------------

    Lg=(ssp.csr_matrix.transpose(DX)@AX@DX)+(ssp.csr_matrix.transpose(DY)@AY@DY)

    #---------------------------------------------------------------
    # Reconstruction of the image
    #---------------------------------------------------------------

    H=Id+(lbda*Lg)

    New_Img_R=ssp.linalg.spsolve(H,np.transpose(img_vec[:,:,0]))
    New_Img_G=ssp.linalg.spsolve(H,np.transpose(img_vec[:,:,1]))
    New_Img_B=ssp.linalg.spsolve(H,np.transpose(img_vec[:,:,2]))

    New_Img=np.zeros([1,nbr_pix,3])
    New_Img[:,:,0]=(1/255)*New_Img_R
    New_Img[:,:,1]=(1/255)*New_Img_G
    New_Img[:,:,2]=(1/255)*New_Img_B

    New_Img=New_Img.reshape([row,col,3])
    # MatX = ssp.csr_matrix.toarray(DX)
    # Maty = ssp.csr_matrix.toarray(DY)
    # MatAX = ssp.csr_matrix.toarray(AX)
    # MatAY = ssp.csr_matrix.toarray(AY)
    # r = Lg@Id@ax@ay@Lg
    return New_Img,ax,ay

#---------------------------------------------------------------

img = extraction_image("../data/Test1.png")
img=img[:,:,:3]   #Use only for image "test1.png"
# img=np.array([[[5,5,3],[4,5,6],[25,25,255]],
            #   [[2,5,3],[4,5,6],[5,55,25]],
            #   [[71,4,3],[4,54,6],[255,25,255]],
            #   [[15,12,3],[47,5,6],[25,55,55]],
            #   [[48,16,3],[41,5,6],[25,25,5]]])


New_Img,ax,ay=WLSFilter(epsilon, alpha, lbda, img)

# print(ssp.csr_matrix.toarray(DX))
# print("\n")
# print(ssp.csr_matrix.toarray(DY))
# print("\n")


ax1=plt.subplot(122)
ax1.imshow(New_Img)
plt.title("Smoothed Image")
plt.show()

plt.subplot(121,sharex=ax1, sharey=ax1)
plt.imshow(img)
plt.title("Original Image")
plt.show()

end = time.time()
print(end - start)
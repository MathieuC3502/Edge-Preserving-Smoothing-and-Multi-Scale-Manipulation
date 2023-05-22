# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:31:04 2023

@author: Mathieu Chesneau
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as ssp

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

def deriveur_x(n):
    """
    Renvoie une matrice de taille n*n qui permet de dériver un vecteur de taille n
    """
    matrice = np.zeros((n, n))
    for i in range(n):
        matrice[i][i] = 2
        if i != n - 1:
            matrice[i][i + 1] = -1
        if i != 0:
            matrice[i][i - 1] = -1
    return matrice

def deriveur_y(n):
    return (deriveur_x(n).transpose)

#---------------------------------------------------------------

# Definition of the parameters of the problem

epsilon=0.0001
alpha=1.6
lbda=10

#---------------------------------------------------------------

img = extraction_image("../data/falaise.jpg")

col = img.shape[1]
row = img.shape[0]
nbr_pix=col*row

Y=np.array(luminance(img))      # Luminance plane of the image
l=np.zeros(Y.shape)             # Log-Luminance plane of the image

for i in range(row):
    for j in range(col):
        l[i,j]=np.log(Y[i,j])   # Log-Luminance plane of the image
        
        
#---------------------------------------------------------------
# Calculation of the ax coefficients
#---------------------------------------------------------------

ax=np.zeros([row,col,3])
ay=np.zeros([row,col,3])     
alpha=1.2
epsilon=0.0001

for i in range(1,row-1):
    for j in range(1,col-1):
        ax[i,j,0]=((abs(2*(img[i,j,0])-img[i-1,j,0]-img[i+1,j,0])**alpha)+epsilon)**-1
        ay[i,j,0]=((abs(2*(img[i,j,0])-img[i,j-1,0]-img[i,j-1,0])**alpha)+epsilon)**-1
        
        ax[i,j,1]=((abs(2*(img[i,j,1])-img[i-1,j,1]-img[i+1,j,1])**alpha)+epsilon)**-1
        ay[i,j,1]=((abs(2*(img[i,j,1])-img[i,j-1,1]-img[i,j-1,1])**alpha)+epsilon)**-1
        
        ax[i,j,2]=((abs(2*(img[i,j,2])-img[i-1,j,2]-img[i+1,j,2])**alpha)+epsilon)**-1
        ay[i,j,2]=((abs(2*(img[i,j,2])-img[i,j-1,2]-img[i,j-1,2])**alpha)+epsilon)**-1
        
ax_vec=ax.reshape(1,nbr_pix,3)
ay=np.transpose(ay)
ay_vec=ay.reshape(1,nbr_pix,3)
img_vec=img.reshape(1,nbr_pix,3)

#---------------------------------------------------------------
# Generation of the Dx matrix
#---------------------------------------------------------------

main_diag=np.ones([1,col-1])
side_diag=-1*(np.ones([1,col-1]))
diagonals=np.array([main_diag,side_diag])
base=ssp.diags(diagonals,[0,1],shape=(col-1,col))

DX=base;

#A way to add a row of zeros at the bottom of a sparse matrix
#Source : https://stackoverflow.com/questions/4695337/expanding-adding-a-row-or-column-a-scipy-sparse-matrix
DX=ssp.csr_matrix(DX)
DX._shape = (DX.shape[0]+1,DX.shape[1])
DX.indptr = np.hstack((DX.indptr,DX.indptr[-1]))

for i in range(row-1):
    DX=ssp.bmat([[DX,None],[None,base]])
    DX=ssp.csr_matrix(DX)
    DX._shape = (DX.shape[0]+1,DX.shape[1])
    DX.indptr = np.hstack((DX.indptr,DX.indptr[-1]))

#---------------------------------------------------------------
# Generation of the Dy matrix
#---------------------------------------------------------------

main_diag=np.ones([1,col-1])
side_diag=-1*(np.ones([1,col-1]))
diagonals=np.array([main_diag,side_diag])
base=ssp.diags(diagonals,[0,1],shape=(col-1,col))

DY=base;

#A way to add a row of zeros at the bottom of a sparse matrix
#Source : https://stackoverflow.com/questions/4695337/expanding-adding-a-row-or-column-a-scipy-sparse-matrix
DY=ssp.csr_matrix(DY)
DY._shape = (DY.shape[0]+1,DY.shape[1])
DY.indptr = np.hstack((DY.indptr,DY.indptr[-1]))

for i in range(row-1):
    DY=ssp.bmat([[DY,None],[None,base]])
    DY=ssp.csr_matrix(DY)
    DY._shape = (DY.shape[0]+1,DY.shape[1])
    DY.indptr = np.hstack((DY.indptr,DY.indptr[-1]))

#---------------------------------------------------------------
# Generation of the AX and AY matrixes
#---------------------------------------------------------------

AXR=ssp.diags(ax_vec[:,:,0],[0])
AYR=ssp.diags(ay_vec[:,:,0],[0])

AXG=ssp.diags(ax_vec[:,:,1],[0])
AYG=ssp.diags(ay_vec[:,:,1],[0])

AXB=ssp.diags(ax_vec[:,:,2],[0])
AYB=ssp.diags(ay_vec[:,:,2],[0])

lbda=2
Id=ssp.identity(AXB.shape[1])

#---------------------------------------------------------------
# Generation of the Lg matrix
#---------------------------------------------------------------

LgR=(ssp.csr_matrix.transpose(DX)@AXR@DX)+(ssp.csr_matrix.transpose(DY)@AYR@DY)
LgG=(ssp.csr_matrix.transpose(DX)@AXG@DX)+(ssp.csr_matrix.transpose(DY)@AYG@DY)
LgB=(ssp.csr_matrix.transpose(DX)@AXB@DX)+(ssp.csr_matrix.transpose(DY)@AYB@DY)

#---------------------------------------------------------------
# Reconstruction of the image
#---------------------------------------------------------------

HR=Id+(lbda*LgR)
HG=Id+(lbda*LgG)
HB=Id+(lbda*LgB)

New_Img_R=ssp.linalg.spsolve(HR,np.transpose(img_vec[:,:,0]))
New_Img_G=ssp.linalg.spsolve(HG,np.transpose(img_vec[:,:,1]))
New_Img_B=ssp.linalg.spsolve(HB,np.transpose(img_vec[:,:,2]))

New_Img=np.zeros([1,nbr_pix,3])
New_Img[:,:,0]=New_Img_R
New_Img[:,:,1]=New_Img_G
New_Img[:,:,2]=New_Img_B

New_Img=New_Img.reshape([row,col,3])

plt.imshow(New_Img)
plt.show()
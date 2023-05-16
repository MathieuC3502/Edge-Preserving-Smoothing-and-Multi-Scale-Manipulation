# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:31:04 2023

@author: Mathieu Chesneau
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

Y=np.array(luminance(img))  # Luminance plane of the image
l=np.zeros(Y.shape)         # Log-Luminance plane of the image

for i in range(row):
    for j in range(col):
        l[i,j]=np.log(Y[i,j])
        
img_vec=img.reshape(1,nbr_pix,3)

Dx=deriveur_x(nbr_pix)


# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:31:04 2023

@author: Mathieu Chesneau
"""

import matplotlib.pyplot as plt
import time
import Functions_carre as Functions
import numpy as np

#---------------------------------------------------------------
# LOADING OF THE DIFFERENT TEST IMAGES
#---------------------------------------------------------------

img1 = Functions.extraction_image("../data/Test1.png")
img1=img1[:,:,:3]
img2 = Functions.extraction_image("../data/landscape.jpg")
img3 = Functions.extraction_image("../data/falaise.jpg")
img4 = Functions.small_matrix_test()
img5 = Functions.extraction_image("../data/fala.jpg")
img6 = Functions.extraction_image("../data/test_ez.jpg")
#---------------------------------------------------------------
#---------------------------------------------------------------

start = time.time()

# Definition of the parameters
epsilon=0.0001
alpha=1.6
lbda=3
w_details = 5

# Choice of the image

def transform_to_square(img):
    return img[:, :img.shape[0], :] if img.shape[0] < img.shape[1] else img[:img.shape[1], :, :]

used_image=transform_to_square(img5)

# Computation of the smoothed image
New_Img, Details=Functions.WLSFilter(epsilon, alpha, lbda/255, used_image)

end = time.time()
print(end - start)

#---------------------------------------------------------------
# PLOTS
#---------------------------------------------------------------

ax1=plt.subplot(122)
ax1.imshow(New_Img)
plt.title("Smoothed Image")
plt.show()

plt.subplot(121,sharex=ax1, sharey=ax1)
plt.imshow(used_image)
plt.title("Original Image")

plt.figure("Detailed Picture")
plt.imshow(New_Img + w_details * Details)
plt.title("Detailed Image")
plt.show()
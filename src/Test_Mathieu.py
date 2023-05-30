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
img5_bis = img5[:, :667, :]
#---------------------------------------------------------------
#---------------------------------------------------------------

start = time.time()

# Definition of the parameters
epsilon=0.0001
alpha=1.6
lbda=0.1

# Choice of the image
print(img5_bis.shape)
used_image=img5_bis

# Computation of the smoothed image
New_Img=Functions.WLSFilter(epsilon, alpha, lbda, used_image)

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
plt.show()
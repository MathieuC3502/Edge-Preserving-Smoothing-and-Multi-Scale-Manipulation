import matplotlib.pyplot as plt
import time
import Functions
import numpy as np

#---------------------------------------------------------------
# LOADING OF THE DIFFERENT TEST IMAGES
#---------------------------------------------------------------

img1 = Functions.extraction_image("../data/Test1.png")
img1=img1[:,:,:3]
img2 = Functions.extraction_image("../data/landscape.jpg")
img3 = Functions.extraction_image("../data/falaise.jpg")
img4 = Functions.small_matrix_test()
img5 = Functions.extraction_image("../data/desert.jpg")
img6 = Functions.extraction_image("../data/test_50x50.jpg")
img7 = Functions.extraction_image("../data/depth_sea.jpg")
img8 = Functions.extraction_image("../data/forest.jpg")
img9 = Functions.extraction_image("../data/city.jpg")
img10 = Functions.extraction_image("../data/canyon.jpg")
img11 = Functions.extraction_image("../data/mountains.jpg")
img7_cut = Functions.extraction_image("../data/depth_sea_cut.jpg")
img8_cut = Functions.extraction_image("../data/forest_cut.jpg")
img9_cut = Functions.extraction_image("../data/city_cut.jpg")
img10_cut = Functions.extraction_image("../data/canyon_cut.jpg")
img11_cut = Functions.extraction_image("../data/mountains_cut.jpg")

"""
Parameters:
- epsilon: float, default = 0.0001
    The threshold for the convergence of the algorithm
- iter: int, default = 3
    The number of iterations
- alpha: float or list of float, default = 1.8, can be a list if iterations (ex: for 3 iterations, in a list, alpha = [1, 2, 1.6])
    The different alpha values for the different iterations
- lbda: float or list of float, default = 0.35, can be a list if iterations (ex: for 3 iterations, in a list, lambda = [1, 2, 1.6])
    The lambda value for the algorithm
- weight: float, list of float or list of list of float, default = 1.2 (can be a list if there are iterations, like: weight = [2, 3, 5] for 3 iterations, and weight each details of iteration)
    Linked to the display parameter
    The weight for the details in iterations
- display: str, default = "smoothed detailed full smoothed full final 3"
    The different images to display
    "smoothed": the smoothed image  
    "detailed": the detailed image
    "full": the full option show all iterations of detail or smoothed add with a space after detailed or smoothed
    "final": the final image, with weights and X final images if add a int after. For ex: affichage = "final 3", means we want 3 final ones. We can change weights of each one by using weight = [[3, 2], [1.2, 0], [0.2, 5]] for example if we want 3 images, in 2 iterations. 
    for "final 2" with iter = 3 for ex can also be weights = [1.2, 3] eqivlent to [[1.2, 1.2, 1.2], [3, 3, 3]] or [2, [2, 3, 0.1]] eqivalent to [[2, 2, 2], [2, 3, 0.1]]
- comparison: str, default = "None"
    The filtered images to compare with the comapared one, but currently only the original image is available, need to add the comparison to gaussian or BLF filters

Also possible to use the class ParamClassFilter

Here are examples below:    
"""

#---------------------------------------------------------------
# TEST 1
#---------------------------------------------------------------
epsilon1 = 0.0001
alpha1 = [1.8, 1.2, 1.6]
iter1 = 3
lbda1 = 0.35
display1 = "smoothed detailed full final 3"
comparaison1 = "original"
weight1 = [[1.2, 0, 1], [0, 5, 10], [5, 5, 4]]
curr_img1 = img1
paramFilter1 = Functions.ParamClassFilter(iter1, alpha1, lbda1, weight1, display1, comparaison1, epsilon1)

#---------------------------------------------------------------
# TEST 2
#---------------------------------------------------------------
epsilon2 = 0.0001
alpha2 = 1.5
iter2 = 5
lbda2 = 0.3
display2 = "final 4"
comparaison2 = None
weight2 = [1.2, [0, 5, 10, 0, 0], [3, 2, 5, 5, 4], 5]
curr_img2 = Functions.limit_size(img9_cut, 300)
paramFilter2 = Functions.ParamClassFilter(iter2, alpha2, lbda2, weight2, display2, comparaison2, epsilon2)

#---------------------------------------------------------------
# TEST 3
#---------------------------------------------------------------

epsilon3 = 0.0001
alpha3 = 1.5
iter3 = 1
lbda3 = 0.3
display3 = "final"
comparaison3 = None
weight3 = 1.3
curr_img3 = Functions.limit_size(img10_cut, 300)
paramFilter3 = Functions.ParamClassFilter(iter3, alpha3, lbda3, weight3, display3, comparaison3, epsilon3)
#---------------------------------------------------------------
# TEST 4
#---------------------------------------------------------------
epsilon4 = 0.0001
alpha4 = 2
iter4 = 3
lbda4 = 0.3
display4 = "smoothed detailed full"
comparaison4 = None
weight4 = 1.5
curr_img4 = Functions.limit_size(img8_cut, 600)
paramFilter4 = Functions.ParamClassFilter(iter4, alpha4, lbda4, weight4, display4, comparaison4, epsilon4)

#---------------------------------------------------------------
# MAIN LOOP
#---------------------------------------------------------------

if __name__ == "__main__":
    # Functions.WLS_full(curr_img1, iter = iter1, alpha = alpha1, lbda = lbda1, weight = weight1, display = display1, comparaison = comparaison1, epsilon = epsilon1)
    Functions.WLS_full(curr_img1, param = paramFilter1)

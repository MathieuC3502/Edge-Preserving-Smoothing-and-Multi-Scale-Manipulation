import matplotlib.pyplot as plt
import time
import Clean_Functions as Functions

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

if __name__ == "__main__":

    # Definition of the parameters
    epsilon = 0.0001
    alpha = 1.6
    iter = 3
    lbda = 3
    # Definition of the images
    curr_img = img2
    # curr_img = Functions.limit_size(img5, 500)

    # Register for differents iterations
    reg_img = []
    reg_det = []

    # List of weigts
    reg_weigt = [0.5, 1, 10]
    start = time.time()

    # Computation of the smoothed image
    if type(lbda) is float or type(lbda) is int:
        list_lbda = [lbda] * iter
    elif len(lbda) == iter:
        list_lbda = lbda.copy()
    else:
        print("Error in the definition of the lambda parameter, lambda set to 2")
        list_lbda = [2] * iter
    for i in range(iter):
        if i == 0:
            smo_img, details = Functions.WLSFilter(epsilon, alpha, list_lbda[i] / 255, curr_img / 255)
        else:
            smo_img, details = Functions.WLSFilter(epsilon, alpha, list_lbda[i] / 255, reg_img[-1])
        reg_img.append(smo_img)
        reg_det.append(details)
        print("For iteration {} it tooks {} seconds".format(i + 1, time.time() - start))

    coa_img = Functions.recreate_img(reg_img[-1], reg_det, reg_weigt)

    # ax1=plt.subplot(132)
    # ax1.imshow(New_Img)
    # plt.title("Smoothed Image")
    # plt.show()

    # plt.subplot(131,sharex=ax1, sharey=ax1)
    # plt.imshow(used_image)
    # plt.title("Original Image")

    # plt.subplot(133,sharex=ax1, sharey=ax1)
    # plt.imshow(New_Img + w_details * Details)
    # plt.title("Detailed Image")
    # plt.show()

    plt.figure("Original")
    plt.imshow(curr_img)
    plt.title("Original Image")
    for i, img in enumerate(reg_img):
        plt.figure("Smoothed {}".format(i + 1))
        plt.imshow(img)
        plt.title("Smoothed Image after iteration {}".format(i + 1))

    plt.figure("Coarsed")
    plt.imshow(coa_img)
    plt.title("Coarsed Image")

    plt.show()
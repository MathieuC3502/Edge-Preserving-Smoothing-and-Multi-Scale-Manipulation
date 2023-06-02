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
img7_cut = Functions.extraction_image("../data/depth_sea_cut.jpg")
img8_cut = Functions.extraction_image("../data/forest_cut.jpg")
img9_cut = Functions.extraction_image("../data/city_cut.jpg")
img10_cut = Functions.extraction_image("../data/canyon_cut.jpg")
img11_cut = Functions.extraction_image("../data/mountains_cut.jpg")

#---------------------------------------------------------------
# MAIN LOOP
#---------------------------------------------------------------

if __name__ == "__main__":

    # Definition of the parameters
    epsilon = 0.0001
    alpha = 2
    iter = 1
    lbda = 6
    # lbda = [2, 4, 8, 16]
    # Definition of the images
    curr_img = Functions.limit_size(img5, 500)

    # Register for differents iterations
    reg_img = []
    reg_det = []

    # List of weights
    # reg_weight = [3, 0, 0, 0]
    reg_weight_0 = 0
    reg_weight_1 = 0.5
    reg_weight_2 = 1
    reg_weight_2a = 1.5
    reg_weight_3 = 2
    reg_weight_4 = 4
    reg_weight_5 = 8
    reg_weight_6 = 16
    start = time.time()

   # Computation of the smoothed image

    list_lbda = Functions.select_param(lbda, iter)
    list_alpha = Functions.select_param(alpha, iter)
    for i in range(iter):
        if i == 0:
            smo_img, details = Functions.WLS_iteration(epsilon, list_alpha[i], list_lbda[i] / 255, curr_img / 255)
        else:
            smo_img, details = Functions.WLS_iteration(epsilon, list_alpha[i], list_lbda[i] / 255, reg_img[-1])
        reg_img.append(smo_img)
        reg_det.append(details)
        print("For iteration {} it tooks {} seconds".format(i + 1, time.time() - start))

    # Computation of the coarsed image after iterations
    # fin_img = Functions.recreate_img(reg_img[-1], reg_det, reg_weight)

    # Computation of the coarsed image at defferent weight
    fin_img_0 = Functions.recreate_img(reg_img[-1], reg_det, reg_weight_0)
    fin_img_1 = Functions.recreate_img(reg_img[-1], reg_det, reg_weight_1)
    fin_img_2 = Functions.recreate_img(reg_img[-1], reg_det, reg_weight_2)
    fin_img_2a = Functions.recreate_img(reg_img[-1], reg_det, reg_weight_2a)
    fin_img_3 = Functions.recreate_img(reg_img[-1], reg_det, reg_weight_3)
    fin_img_4 = Functions.recreate_img(reg_img[-1], reg_det, reg_weight_4)
    fin_img_5 = Functions.recreate_img(reg_img[-1], reg_det, reg_weight_5)
    fin_img_6 = Functions.recreate_img(reg_img[-1], reg_det, reg_weight_6)

    end_time = time.time() - start
#---------------------------------------------------------------
# PLOTS
#---------------------------------------------------------------
    # plt.figure()

    # ax1=plt.subplot(121)
    # plt.imshow(curr_img)
    # plt.title("Original Image")
    
#---------------------------------------------------------------
    
    # plt.figure()
    
    ### To plot smoothed image and detailed images for each iteration
    # for i, img in enumerate(reg_img):
    #     plt.subplot(iter, 2, 2 * i + 1)
    #     plt.imshow(img)
    #     plt.title("Smoothed Image after iteration {}".format(i + 1))
    #     plt.subplot(iter, 2, 2 * i + 2)
    #     plt.imshow(40 * (0.299 * reg_det[i][:, :, 0] +  0.587 * reg_det[i][:, :, 1] + 0.119 * reg_det[i][:, :, 2]), cmap = 'gray')
    #     plt.title("Details of Image after iteration {}".format(i + 1))

    # To plot smoothed images with 4 iterations only 

    # for i in range(4):
    #     plt.subplot(2,2,i+1)
    #     plt.imshow(reg_img[i])
    #     plt.title("Lambda = {}, Alpha = {}".format(list_lbda[i], list_alpha[i]))
    
    # To plot only smoothed images alone
    # for i in range(iter):
    #     plt.figure("Image {}".format(i + 4))
    #     plt.imshow(reg_img[i])
    #     plt.title("Lambda = {}, Alpha = {}".format(list_lbda[i], list_alpha[i]))
    # plt.figure()

    # To plot details of images with 4 iterations only 
    # for i in range(4):
    #     plt.subplot(2,2,i+1)
    #     plt.imshow(40 * (0.299 * reg_det[i][:, :, 0] +  0.587 * reg_det[i][:, :, 1] + 0.119 * reg_det[i][:, :, 2]), cmap = 'gray')
    #     plt.title("Lambda = {}, Alpha = {}".format(list_lbda[i], list_alpha[i]))
    
    # To plot only details of images alone
    # for i in range(4):
    #     plt.figure("Image {}".format(i))
    #     plt.imshow(40 * (0.299 * reg_det[i][:, :, 0] +  0.587 * reg_det[i][:, :, 1] + 0.119 * reg_det[i][:, :, 2]), cmap = 'gray')
    #     plt.title("Lambda = {}, Alpha = {}".format(list_lbda[i], list_alpha[i]))
    
    # To look at timings 
    # plt.figure("Time of running")
    # plt.imshow(reg_img[i])
    # plt.title("Lambda: {}, Alpha: {}, Time: {}s, Shape = {}x{}".format(list_lbda[i], list_alpha[i], int(100 *end_time) / 100, shape[0], shape[1]))
    
    # To look at images with details 
    plt.figure("Final Image 0")
    plt.imshow(fin_img_0)
    plt.title("Final Image 0")
    plt.figure("Final Image 0.5")
    plt.imshow(fin_img_1)
    plt.title("Final Image 0.5")
    plt.figure("Final Image 1")
    plt.imshow(fin_img_2)
    plt.title("Final Image 1")
    plt.figure("Final Image 1.5")
    plt.imshow(fin_img_2a)
    plt.title("Final Image 1.5")
    plt.figure("Final Image 2")
    plt.imshow(fin_img_3)
    plt.title("Final Image 2")
    plt.figure("Final Image 4")
    plt.imshow(fin_img_4)
    plt.title("Final Image 4")
    plt.figure("Final Image 8")
    plt.imshow(fin_img_5)
    plt.title("Final Image 8")
    plt.figure("Final Image 16")
    plt.imshow(fin_img_6)
    plt.title("Final Image 16")
    plt.show()
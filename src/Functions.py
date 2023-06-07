import numpy as np
import scipy.sparse as ssp
from PIL import Image
import time
import matplotlib.pyplot as plt

class ParamClassFilter:
    def __init__(self, iter: int = 1, alpha: float = 1.8, lbda: float = 0.3, weight: float = 1.2, display: str = "smoothed full", comparaison = None, epsilon: float = 0.00001) -> None:
        self.iter = iter
        self.alpha = alpha
        self.lbda = lbda
        self.weight = weight
        self.display = display
        self.comparaison = comparaison
        self.epsilon = epsilon

def small_matrix_test():
    img=np.zeros((3,4,3))
    img[:,:,0]=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    img[:,:,1]=np.array([[0,0,0,0],[255,0,0,255],[0,0,0,0]])
    img[:,:,2]=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    return img

def select_param(param, iter):
    if type(param) is float or type(param) is int:
        list_param = [param] * iter
        print("Parameter set to {}".format(list_param))
    elif len(param) == iter:
        list_param = param.copy()
        print("Parameter set to {}".format(list_param))
    else:
        list_param = [1.6] * iter
        print("Error in the definition of the parameter, set to {}".format(list_param))
    return list_param

def transform_to_square(img):
    return img[:, :img.shape[0], :] if img.shape[0] < img.shape[1] else img[:img.shape[1], :, :]

def limit_size(img, pix):
    if img.shape[0] > pix:
        img = img[:pix, :, :]
    if img.shape[1] > pix:
        img = img[:, :pix + 200, :]
    return img

def recreate_img(smo_img, list_of_details, list_of_weights):
    final_img = smo_img.copy()
    for i in range(len(list_of_details)):
        final_img += list_of_weights[i] * list_of_details[i]
    return final_img

def extraction_image(nomFichier):
    img = Image.open(nomFichier)
    return np.array(img)

def luminance(img):
    return (.299 * img[:,:,0]) + (.587 * img[:,:,1]) + (.114 * img[:,:,2])


# Here are the old versions of the matrices, without removing zeros, and edge effects. 
def DX_matrix(nbr_pix):
    diagonals = [np.ones(nbr_pix).T, - np.ones(nbr_pix - 1).T]
    return ssp.csr_matrix(ssp.diags(diagonals, [0, 1]))

def DY_matrix(row, col):
    diagonals = [np.concatenate((np.ones((col - 1) * row).T, np.zeros(row).T)), - np.ones((row - 1) * col)]
    return ssp.csr_matrix(ssp.diags(diagonals, [0, col]))

def ax_coeffs(row, col, logL, alpha, epsilon):
    diff = np.absolute(np.concatenate((logL[:, :col - 1] - logL[:, 1:], np.zeros((row, 1))), axis = 1))
    ax = 1 / (np.power(diff, alpha) + (epsilon * np.ones((row, col))))
    return ax.reshape(1, row * col)

def ay_coeffs(row, col, logL, alpha, epsilon):
    diff = np.absolute(np.concatenate((logL[:row - 1, :] - logL[1:, :], np.zeros((1, col))), axis = 0))
    ay = 1 / (np.power(diff, alpha) + (epsilon * np.ones((row, col))))
    return ay.reshape(1, row * col)


# Here are the new matrices, finally with a little size, winning some time of calculation and avoiding edge effects
def DX_matrix_reduced(row, col):
    main_diag=np.ones([1,col-1])
    side_diag=-1*(np.ones([1,col-1]))
    diagonals=np.array([main_diag,side_diag])
    base=ssp.diags(diagonals,[0,1],shape=(col-1,col))
    DX=base
    DX=ssp.csr_matrix(DX)
    for i in range(row-1):
        DX=ssp.bmat([[DX,None],[None,base]])
        DX=ssp.csr_matrix(DX)

    return DX

def DY_matrix_reduced(row, col):
    diagonals = [np.ones((row - 1) * col), - np.ones((row - 1) * col)]
    return ssp.csr_matrix(ssp.diags(diagonals, [0, col], shape=((row - 1) * col, row * col)))


def ax_coeffs_reduced(row, col, logL, alpha, epsilon):
    diff = np.absolute(logL[:, :col - 1] - logL[:, 1:])
    ax = 1 / (np.power(diff, alpha) + (epsilon * np.ones((row, col - 1))))
    return ax.reshape(1, row * (col - 1))

def ay_coeffs_reduced(row, col, logL, alpha, epsilon):
    diff = np.absolute(logL[:row - 1, :] - logL[1:, :])
    ay = 1 / (np.power(diff, alpha) + (epsilon * np.ones((row - 1, col))))
    return ay.reshape(1, (row - 1) * col)

def WLS_iteration(epsilon,alpha,lbda,img):

    (row, col, RGB) = img.shape
    nbr_pix = col * row
    logL = np.log(luminance(img) + np.ones((row, col)))   # Log-Luminance plane of the image

    ax_vec = ax_coeffs_reduced(row, col, logL, alpha, epsilon)
    ay_vec = ay_coeffs_reduced(row, col, logL, alpha, epsilon)
    img_vec = img.reshape(1, nbr_pix, 3)
          
    DX = DX_matrix_reduced(row, col)
    DY = DY_matrix_reduced(row, col)

    AX = ssp.diags(ax_vec,[0])
    AY = ssp.diags(ay_vec,[0])
    Lg = ssp.csr_matrix.transpose(DX)@AX@DX + ssp.csr_matrix.transpose(DY)@AY@DY 
    Id = ssp.identity(DY.shape[1])

    #---------------------------------------------------------------
    # Reconstruction of the image
    #---------------------------------------------------------------

    H = Id + (lbda * Lg)
    New_Img = np.zeros((1, nbr_pix, 3))

    New_Img[:,:,0] = ssp.linalg.spsolve(H, np.transpose(img_vec[:,:,0]))

    New_Img[:,:,1] = ssp.linalg.spsolve(H, np.transpose(img_vec[:,:,1]))

    New_Img[:,:,2] = ssp.linalg.spsolve(H, np.transpose(img_vec[:,:,2]))

    New_Img = New_Img.reshape((row, col, 3))

    return New_Img, (img - New_Img)

def WLS_full(img, iter = 1, alpha = 1.8, lbda = 0.35, weight = 1.2, display = "smoothed full", comparaison = None, epsilon = 0.0001, param: ParamClassFilter = None):
    if param is not None:
        iter = param.iter
        alpha = param.alpha
        lbda = param.lbda
        weight = param.weight
        display = param.display
        comparaison = param.comparaison
        epsilon = param.epsilon
    reg_img = []
    reg_det = []
    start = time.time()
    print("Set of the lambda parameter")
    list_lbda = select_param(lbda, iter)
    print("Set of the alpha parameter")
    list_alpha = select_param(alpha, iter)
    for i in range(iter):
        print("Run of the iteration {}".format(i + 1))
        if i == 0:
            smo_img, details = WLS_iteration(epsilon, list_alpha[i], list_lbda[i] / 10, img / 255)
        else:
            smo_img, details = WLS_iteration(epsilon, list_alpha[i], list_lbda[i] / 10, reg_img[-1])
        reg_img.append(smo_img)
        reg_det.append(details)
        print("For iteration {} it tooks {} seconds".format(i + 1, time.time() - start))

    list_aff = display.split()
    done = False

    if "final" in list_aff:
        final_img = []
        ind = list_aff.index("final")
        if len(list_aff) > ind + 1:
            try:
                num_show = int(list_aff[ind + 1])
                for i in range(num_show):
                    print("Set of the {} weight parameter".format(i + 1))
                    list_weight = select_param(weight[i], iter)
                    final_img.append(recreate_img(reg_img[-1], reg_det, list_weight))
                    plt.figure("Final Image {}, weights per iterations: {}".format(i + 1, list_weight))
                    plt.imshow(final_img[-1])
                    plt.title("Final Image {}, weights per iterations: {}".format(i + 1, list_weight))
                print("Show all final images, additionning wieghted details to smoothed picture")
                
            except:
                if iter == 1:
                    print("Show the final image, additionning wieghted details to smoothed picture")
                else:
                    print("Weigts are unclear for each final picture, so one picture is resulting with maybe the wrong weights applied")
                print("Set of the weight parameter")
                list_weight = select_param(weight, iter)
                final_img.append(recreate_img(reg_img[-1], reg_det, list_weight))
                plt.figure("Final Image, weights per iterations: {}".format(list_weight))
                plt.imshow(final_img[-1])
                plt.title("Final Image, weights per iterations: {}".format(list_weight))

    if "smoothed" in list_aff:
        ind = list_aff.index("smoothed")
        if len(list_aff) > ind + 1:
            if list_aff[ind + 1] == "full":
                print("Show all the smoothed images during iterations")
                for i in range(iter):
                    plt.figure("Smoothed Image for iteration {}".format(i + 1))
                    plt.imshow(reg_img[i])
                    plt.title("Lambda = {}, Alpha = {}".format(list_lbda[i], list_alpha[i]))
            elif list_aff[ind + 1] == "detailed":
                done = True
                if len(list_aff) > ind + 2:
                    if list_aff[ind + 2] == "full":
                        print("Show all the smoothed images and details during iterations")
                        for i in range(len(reg_img)):
                            plt.figure("Results for iteration {}".format(i + 4))
                            plt.subplot(1, 2, 1)
                            plt.imshow(reg_img[i])
                            plt.title("Smoothed Image after iteration {}".format(i + 1))
                            plt.subplot(1, 2, 2)
                            plt.imshow(40 * (0.299 * reg_det[i][:, :, 0] +  0.587 * reg_det[i][:, :, 1] + 0.119 * reg_det[i][:, :, 2]), cmap = 'gray')
                            plt.title("Details of Image after iteration {}".format(i + 1))
                    else:
                        print("Show the smoothed images and details obtained")
                        plt.subplot(1, 2, 1)
                        plt.imshow(reg_img[-1])
                        plt.title("Last Smoothed Image")
                        plt.subplot(1, 2, 2)
                        plt.imshow(40 * (0.299 * reg_det[-1][:, :, 0] +  0.587 * reg_det[-1][:, :, 1] + 0.119 * reg_det[-1][:, :, 2]), cmap = 'gray')
                        plt.title("Last Details of Image")
                        
            else:
                print("Show the last smoothed image")
                plt.figure("Last Smoothed Image")
                plt.imshow(reg_img[i])
                plt.title("Lambda = {}, Alpha = {}".format(list_lbda[-1], list_alpha[-1]))

    if "detailed" in list_aff and not done:
        ind = list_aff.index("detailed")
        if len(list_aff) > ind + 1:
            if list_aff[ind +1] == "full":
                print("Show all the details during iterations")
                for i in range(len(reg_det)):
                    plt.figure("Detailed Image for iteration {}".format(i + 4))
                    plt.imshow(40 * (0.299 * reg_det[i][:, :, 0] +  0.587 * reg_det[i][:, :, 1] + 0.119 * reg_det[i][:, :, 2]), cmap = 'gray')
                    plt.title("Lambda = {}, Alpha = {}".format(list_lbda[i], list_alpha[i]))
            else:
                print("Show the last details obtained")
                plt.figure("Last Detailed Image")
                plt.imshow(40 * (0.299 * reg_det[i][:, :, 0] +  0.587 * reg_det[i][:, :, 1] + 0.119 * reg_det[i][:, :, 2]), cmap = 'gray')
                plt.title("Lambda = {}, Alpha = {}".format(list_lbda[i], list_alpha[i]))
    if comparaison == "original":
        print("Comparison to the original version")
        plt.figure("Original Image")
        plt.imshow(img)
        plt.title("Original Image")
    plt.show()
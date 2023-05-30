import numpy as np
import scipy.sparse as ssp
from PIL import Image
import time

def small_matrix_test():
    img=np.zeros((3,4,3))
    img[:,:,0]=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    img[:,:,1]=np.array([[0,0,0,0],[255,0,0,255],[0,0,0,0]])
    img[:,:,2]=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    return img

def transform_to_square(img):
    return img[:, :img.shape[0], :] if img.shape[0] < img.shape[1] else img[:img.shape[1], :, :]

def limit_size(img, pix):
    return img[:pix, :pix, :]

def recreate_img(smo_img, list_of_details, weigts = None):
    final_img = smo_img.copy()
    if weigts is None:
        print("Weigts all set to 1.2")
        weigts = [1.2 for i in range(len(list_of_details))]
    elif type(weigts) is float or type(weigts) is int:
        print("Weigts set to {}".format(weigts))
        weigts = [weigts for i in range(len(list_of_details))]
    elif len(weigts) != len(list_of_details):
        print("A mistake has been Done, retry to compute the right number of weigts, weigts set by default to 1.2")
        weigts = [1.2 for i in range(len(list_of_details))]
    elif len(weigts) == len(list_of_details): 
        print("Weigts set to {}".format(weigts))
    else:
        print("Weigts set to 1.2 due to unknown variable")
        weigts = [1.2 for i in range(len(list_of_details))]
    for i in range(len(list_of_details)):
        final_img += weigts[i] * list_of_details[i]
    return final_img

def extraction_image(nomFichier):
    img = Image.open(nomFichier)
    return np.array(img)

def luminance(img):
    return (.299 * img[:,:,0]) + (.587 * img[:,:,1]) + (.114 * img[:,:,2])

def DX_matrix(nbr_pix):
    diagonals = [np.ones(nbr_pix).T, - np.ones(nbr_pix - 1).T]
    return ssp.csr_matrix(ssp.diags(diagonals, [0, 1]))

def DY_matrix(row, col):
    diagonals = [np.concatenate((np.ones((col - 1) * row).T, np.zeros(row).T)), - np.ones((col - 1) * row)]
    return ssp.csr_matrix(ssp.diags(diagonals, [0, col]))

def ax_coeffs(row, col, logL, alpha, epsilon):
    diff = np.absolute(np.concatenate((logL[:, :col - 1] - logL[:, 1:], np.zeros((row, 1))), axis = 1))
    ax = 1 / (np.power(diff, alpha) + (epsilon * np.ones((row, col))))
    return ax.reshape(1, row * col)

def ay_coeffs(row, col, logL, alpha, epsilon):
    diff = np.absolute(np.concatenate((logL[:row - 1, :] - logL[1:, :], np.zeros((1, col))), axis = 0))
    ay = 1 / (np.power(diff, alpha) + (epsilon * np.ones((row, col))))
    return ay.reshape(1, row * col)


def WLSFilter(epsilon,alpha,lbda,img):

    print("Starting iteration and filtering matrices")

    # print("Computation of Log-Luminance")
    (row, col, RGB) = img.shape
    nbr_pix = col * row
    logL = np.log(luminance(img) + np.ones((row, col)))   # Log-Luminance plane of the image

    # print("Computation of Ax/Ay coefficients")     
    ax_vec = ax_coeffs(row, col, logL, alpha, epsilon)
    ay_vec = ay_coeffs(row, col, logL, alpha, epsilon)
    img_vec = img.reshape(1, nbr_pix, 3)

    # print("Computation of DX Matrix")
    DX = DX_matrix(row * col)

    # print("Computation of DY Matrix")
    DY = DY_matrix(row, col)

    # print("Computation of the AX/AY Matrixes")
    AX = ssp.diags(ax_vec,[0])
    AY = ssp.diags(ay_vec,[0])
    Id = ssp.identity(AX.shape[1])

    # print("Computation of Lg Matrix")
    Lg = ssp.csr_matrix.transpose(DX)@AX@DX + ssp.csr_matrix.transpose(DY)@AY@DY 

    #---------------------------------------------------------------
    # Reconstruction of the image
    #---------------------------------------------------------------

    H = Id + (lbda * Lg)
    New_Img = np.zeros((1, nbr_pix, 3))

    print("Creating the smoothed image")
    New_Img[:,:,0] = ssp.linalg.spsolve(H, np.transpose(img_vec[:,:,0]))

    New_Img[:,:,1] = ssp.linalg.spsolve(H, np.transpose(img_vec[:,:,1]))

    New_Img[:,:,2] = ssp.linalg.spsolve(H, np.transpose(img_vec[:,:,2]))

    print("Filtering Iteration Done")
    New_Img = New_Img.reshape((row, col, 3))

    return New_Img, (img - New_Img)
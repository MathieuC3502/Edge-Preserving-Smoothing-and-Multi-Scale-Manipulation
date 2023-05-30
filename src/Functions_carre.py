import numpy as np
import scipy.sparse as ssp
from PIL import Image

def DX_matrix(row,col):
    nbr_pix = row*col
    main_diag=np.ones((1,nbr_pix))
    side_diag=-1*(np.ones((1,nbr_pix - 1)))
    diagonals=[main_diag,side_diag]
    base=ssp.diags(diagonals,[0,1],shape=(nbr_pix, nbr_pix))
    base=ssp.csr_matrix(base)
    return base

def DY_matrix(row,col):
    nbr_pix=row*col
    ONE = np.ones((nbr_pix-row))
    ZER = np.zeros((row))
    main_diag=np.concatenate((ONE.T, ZER.T))
    side_diag=-1*(np.ones([1,nbr_pix-row]))
    diagonals=[main_diag,side_diag]
    DY=ssp.diags(diagonals,[0,row],shape=(nbr_pix,nbr_pix))
    DY=ssp.csr_matrix(DY)
    return DY

def luminance(img):
    R=img[:,:,0]
    G=img[:,:,1]
    B=img[:,:,2]
    return (.299 * R) + (.587 * G) + (.114 * B)

def extraction_image(nomFichier):
    img = Image.open(nomFichier)
    return np.array(img)

def ax_coeffs(row,col,l,alpha,epsilon):
    derivx=np.zeros((row,col))
    derivx[:,:col-1]=l[:,:col-1]-l[:,1:]
    derivx=np.absolute(derivx)
    ax=np.power(derivx, alpha)
    ax=ax+(epsilon*np.ones([row,col]))
    ax=1/ax
    return ax.reshape(1,row*col)

def ay_coeffs(row,col,l,alpha,epsilon):
    derivy=np.zeros([row,col])
    derivy[:row-1,:]=l[:row-1,:]-l[1:,]
    derivy=np.absolute(derivy)
    ay=np.power(derivy, alpha)
    ay=ay+(epsilon*np.ones([row,col]))
    ay=1/ay
    return ay.reshape(1, row*col)

def AXY_matrix(a_vec):
    A=ssp.diags(a_vec,[0])
    return A

def small_matrix_test():
    img=np.zeros((3,4,3))
    img[:,:,0]=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    img[:,:,1]=np.array([[0,0,0,0],[255,0,0,255],[0,0,0,0]])
    img[:,:,2]=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    return img

def WLSFilter(epsilon,alpha,lbda,img):
    print("Starting")
    # (row, col, RGB) = img.shape
    # nbr_pix=col*row

    # Y=np.array(luminance(img))      # Luminance plane of the image
    # l=np.zeros(Y.shape)             # Log-Luminance plane of the image

    print("Computation of Log-Luminance")
    (row, col, RGB) = img.shape
    nbr_pix=col*row
    l = np.log(luminance(img) + np.ones((row, col)))   # Log-Luminance plane of the image
            
            
    #---------------------------------------------------------------
    # Computation of the ax/ay coefficients
    #---------------------------------------------------------------

    print("Computation of Ax/Ay coefficients")
            
    ax_vec=ax_coeffs(row,col,l,alpha,epsilon)
    ay_vec=ay_coeffs(row,col,l,alpha,epsilon)
    img_vec=img.reshape(1,nbr_pix,3)

    #---------------------------------------------------------------
    # Generation of the Dx matrix
    #---------------------------------------------------------------

    print("Computation of DX Matrix")
    DX=DX_matrix(row, col)

    #---------------------------------------------------------------
    # Generation of the Dy matrix
    #---------------------------------------------------------------

    print("Conputation of DY Matrix")
    DY=DY_matrix(row,col)

    #---------------------------------------------------------------
    # Generation of the AX and AY matrixes
    #---------------------------------------------------------------

    print("Computation of the AX/AY Matrixes")
    AX=AXY_matrix(ax_vec)
    AY=AXY_matrix(ay_vec)

    Id=ssp.identity(AX.shape[1])

    #---------------------------------------------------------------
    # Generation of the Lg matrix
    #---------------------------------------------------------------

    print("Computation of Lg Matrix")
    print(DX.shape)
    print(AX.shape)
    Lg=(ssp.csr_matrix.transpose(DX)@AX@DX)+(ssp.csr_matrix.transpose(DY)@AY@DY)

    #---------------------------------------------------------------
    # Reconstruction of the image
    #---------------------------------------------------------------

    H=Id+(lbda*Lg)

    print("Computation of the New R-Plane")
    New_Img_R=ssp.linalg.spsolve(H,np.transpose(img_vec[:,:,0]))
    print("Computation of the New G-Plane")
    New_Img_G=ssp.linalg.spsolve(H,np.transpose(img_vec[:,:,1]))
    print("Computation of the New B-Plane")
    New_Img_B=ssp.linalg.spsolve(H,np.transpose(img_vec[:,:,2]))

    New_Img=np.zeros([1,nbr_pix,3])
    New_Img[:,:,0]=(1/255)*New_Img_R
    New_Img[:,:,1]=(1/255)*New_Img_G
    New_Img[:,:,2]=(1/255)*New_Img_B

    New_Img=New_Img.reshape([row,col,3])
    # MatX = ssp.csr_matrix.toarray(DX)
    # Maty = ssp.csr_matrix.toarray(DY)
    # AY=ssp.csr_matrix(AY)
    # AX=ssp.csr_matrix(AX)
    # MatAX = ssp.csr_matrix.toarray(AX)
    # MatAY = ssp.csr_matrix.toarray(AY)
    # MatLg = ssp.csr_matrix.toarray(Lg)
    print("Done")
    # a = New_Img @ New_Img
    return New_Img
import numpy as np
import scipy.sparse as ssp
from PIL import Image

def DX_matrix(row,col):
    main_diag=np.ones([1,col-1])
    side_diag=-1*(np.ones([1,col-1]))
    diagonals=np.array([main_diag,side_diag])
    base=ssp.diags(diagonals,[0,1],shape=(col-1,col))

    DX=base

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
        
    return DX

def DY_matrix(row,col):
    nbr_pix=row*col
    main_diag=np.ones([1,nbr_pix])
    side_diag=-1*(np.ones([1,nbr_pix-row]))
    diagonals=[[main_diag],[side_diag]]
    DY=ssp.diags(diagonals,[0,row],shape=(nbr_pix-row,nbr_pix))
    DY=ssp.csr_matrix(DY)

    #A way to add a row of zeros at the bottom of a sparse matrix
    #Source : https://stackoverflow.com/questions/4695337/expanding-adding-a-row-or-column-a-scipy-sparse-matrix
    for i in range(row):
        DY=ssp.csr_matrix(DY)
        DY._shape = (DY.shape[0]+1,DY.shape[1])
        DY.indptr = np.hstack((DY.indptr,DY.indptr[-1]))
        
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
    derivx=np.zeros([row,col])
    derivx[:,:col-1]=l[:,:col-1]-l[:,1:]
    derivx[:,col-1]=np.zeros([1,row])
    derivx=np.absolute(derivx)
    ax=np.power(derivx, alpha)
    ax=ax+(epsilon*np.ones([row,col]))
    ax=1/ax
    return ax.reshape(1,row*col)

def ay_coeffs(row,col,l,alpha,epsilon):
    derivy=np.zeros([row,col])
    derivy[:row-1,:]=l[:row-1,:]-l[1:,]
    derivy[:row-1,:]=np.zeros([1,col])
    derivy=np.absolute(derivy)
    ay=np.power(derivy, alpha)
    ay=ay+(epsilon*np.ones([row,col]))
    ay=1/ay
    return ay.reshape(1,row*col)

def AXY_matrix(a_vec):
    a_vec=np.ones([1,a_vec.shape[1]])
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
    col = img.shape[1]
    row = img.shape[0]
    nbr_pix=col*row

    Y=np.array(luminance(img))      # Luminance plane of the image
    l=np.zeros(Y.shape)             # Log-Luminance plane of the image

    print("Computation of Log-Luminance")
    for i in range(row):
        for j in range(col):
            l[i,j]=np.log(Y[i,j]+1)   # Log-Luminance plane of the image
            
            
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
    
    print("Done")
    return New_Img
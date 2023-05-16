from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

print("Début projet Image")

def extraction_image(nomFichier):
    img = Image.open(nomFichier)
    return np.array(img)

def dt_color(x, y):
    return abs(x - y)

def dt_eucly(xa, ya, xb, yb):
    return np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)

def gaussian_function(x, sig):
    return (1 / (sig * np.sqrt(2 * np.pi))) * np.exp(- (x ** 2) / (2 * sig ** 2))

img = extraction_image("../data/falaise.jpg")


img_3 = np.array(Image.open("../data/falaise.jpg").convert('L'))


col = img.shape[1]
row = img.shape[0]

print("Colonnes :", col, "\n")
print("Lignes :", row, "\n")

sig_s = 4
sig_r = 0.15
img_2 = np.zeros((row, col))
for i in range(col):
    for j in range(row):
        sum = 0
        coeff = 0
        for k in range(col):
            for l in range(row):
                col_p = img_3[j][i]
                col_q = img_3[l][k]
                val = gaussian_function(dt_eucly(i, j, k, l), sig_s) * gaussian_function(dt_color(col_p, col_q), sig_r)
                coeff += val
                sum += val * col_q 
        img_2[j][i] = sum / coeff

                

plt.figure("Image originale")
plt.imshow(img)

plt.figure("Image noir et blanc 2")
plt.imshow(img_2, cmap="gray")

plt.show()

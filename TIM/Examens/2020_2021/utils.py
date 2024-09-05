import matplotlib.pyplot as plt
import numpy as np
import cv2 
from sklearn.cluster import MiniBatchKMeans

# Quantification en niveaux de gris
def imquantize(img,N):
    # Inputs :
    #     img : image en niveaux de gris (uint8)
    #     N : le nombre de niveaux de gris maximum à conserver
    # Output : image quantifiée en N niveau de gris (uint8)
    M = img.max()
    m = img.min()
    pas = (M-m)/(N-1)
    seuils = [np.floor(m+pas*n).astype(np.uint8) for n in range(N)]
    imgQuantif = np.full(img.shape,m)
    for seuil in seuils:
        imgQuantif[img.astype(np.int16)-seuil>0]=seuil
    return imgQuantif.astype(np.uint8)

# Quantification couleur
def imcolorquantize(img,N):
    # Inputs :
    #     img : image couleur (uint8)
    #     N : le nombre de couleurs maximum à conserver
    # Output : image quantifiée en N couleurs (uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    (h, w) = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(N)
    labels = clt.fit_predict(img)
    quant = clt.cluster_centers_.astype("uint8")[labels]
     
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
 
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)

    return quant

# Expansion de la dynamique
def exp_dyn(img,a,b):
    # Inputs :
    #    img : image en niveaux de gris (uint8)
    #    a,b : seuils pour l'expansion (int)
    # Output : image en niveaux de gris (uint8)
    return np.floor(255*np.clip((img.astype(np.int16)-a)/(b-a),0,1))

# Réhaussement de contraste
def reh_const(img,a,b):
    # Inputs :
    #    img : image en niveaux de gris (uint8)
    #    a,b : seuils pour l'expansion (int)
    # Output : image en niveaux de gris (uint8)
    img = img.astype(np.int16)
    new_img = np.empty(img.shape)
    new_img[img<=a]=img[img<=a]*b/a
    new_img[img>a]=((255-b)*img[img>a]+255*(b-a))/(255-a)
    return np.floor(new_img).astype(np.uint8)
                                            
# Bruitage Sel et Poivre
def salt_and_pepper_noise(img,p):
    # Inputs :
    #    img : image en niveaux de gris (uint8)
    #    p : pourcentage de pixels à bruiter
    # Output : image bruitée en niveaux de gris (uint8)
    
    # Création masque aléatoire avec p% de 0 et (100-p)% de 1
    mask=(np.random.randint(100,size=img.shape)>p)
    # Image bruit avec aléatoirement du blanc et du noir
    new_img = np.random.randint(2,size=img.shape)*255
    # Tous les pixels de mask à 0, prennent la valeur de new_img (0 ou 255) dans l'image originale
    new_img[mask]=img[mask]
    return new_img.astype(np.uint8)

# Bruitage Gaussien
def gaussian_noise(img,mean,std):
    # Inputs :
    #    img : image en niveaux de gris (uint8)
    #    mean,std : moyenne et écart-type du bruit gaussien à appliquer
    # Output : image bruitée en niveaux de gris (uint8)
    
    # Création de la matrice de bruit gaussien, puis ajout sur l'image originale.
    # Enfin, on s'assure que les valeurs de notre nouvelle image se situent entre 0 et 255 et on convertit en entier.
    return np.clip(img + np.random.normal(mean, std, img.shape),0,255).astype(np.uint8)

# Sélection de couleur sur image HSV
def HSV_color_selection(image,lower_color,upper_color):
    # Inputs :
    #    image : image couleur RGB (uint8)
    #    lower_color, upper_color : vecteurs numpy de taille 3 indiquant les codes couleurs limites de l'intervalle de couleur à garder / ex : lower_color = np.array([0,0,0]), upper_color = np.array([100,200,50])
    # Outputs :
    #    mask : masque de sélection des pixels sur l'image (bool)
    #    res : masque de sélection des pixels sur l'image, dans leur couleur d'origine (uint8) 
    
    # HSV color conversion
    hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    # Threshold the HSV image to get only colors between lower_color and upper_color
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    # Image finale avec uniquement les pixels du masque sélectionnés (en couleur)
    output = image.copy()
    output[mask==0] = np.array([0,0,0])
    return mask,output
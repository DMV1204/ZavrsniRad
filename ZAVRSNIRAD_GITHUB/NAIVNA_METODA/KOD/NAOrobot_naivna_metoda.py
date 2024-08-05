import sys
import time
from PIL import Image
from naoqi import ALProxy
import cv2
import numpy as np
import skimage.exposure
import matplotlib.pyplot as plt

#pip install matplotlib==2.2.5

#Funkcija za preuzimanje slike s robota
def showNaoImage(IP, PORT):
    camProxy = ALProxy("ALVideoDevice", IP, PORT)
    resolution = 2
    colorSpace = 11

    # Pristupamo kameri metodom subsribeCamera
    videoClient = camProxy.subscribeCamera("python_client", 0, resolution, colorSpace, 5)

    t0 = time.time()
    # Uzimamo sliku sa kamere
    naoImage = camProxy.getImageRemote(videoClient)

    t1 = time.time()

    print("Time delay ", t1 - t0)

    # Odjavljujemo se sa kamere
    camProxy.unsubscribe(videoClient)

    # Uzimamo podatke o slici iz naoImage kojeg je vratila metoda getImageRemote
    imageWidth = naoImage[0]
    imageHeight = naoImage[1]
    array = naoImage[6] # Pikseli slike

    # Jer imamo RAW format, koristimo PIL za konverziju u sliku
    im = Image.frombytes("RGB", (imageWidth, imageHeight), array)

    # Sačuvamo sliku
    image_path = "camImage.png"
    im.save(image_path, "PNG")

    return image_path

#Funkcija za stvaranje slike ruba
def edge_detection(image_path):


    img = cv2.imread(image_path)

    # Obrada slike za Canny detekciju rubova
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    A = lab[:,:,1]

    thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=5, sigmaY=5, borderType=cv2.BORDER_DEFAULT)

    mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255)).astype(np.uint8)

    result = img.copy()
    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result[:,:,3] = mask

    # Prikaz procesa obrade
    cv2.imwrite('greenscreen_thresh.png', thresh)
    cv2.imwrite('greenscreen_mask.png', mask)
    cv2.imwrite('greenscreen_antialiased.png', result)

    # Pozivanje Canny algoritma
    edges = cv2.Canny(mask, 50, 150)

    # Sačuvamo edge-detection sliku
    edges_path = 'edges_' + image_path
    cv2.imwrite(edges_path, edges)

    return edges_path

def process_image_and_check_circle(image_path):
    #Učitavamo edge-detection sliku
    new_image = Image.open(image_path).convert("L")

    # Prebacujemo ju u numpy array
    new_image_array = np.array(new_image)

    # Radimo binarni zapis slike (bijeli pikseli = 1, crni pikseli = 0)
    new_binary_array = np.where(new_image_array > 128, 1, 0)

    # Pohranjujemo koordinate svih bijelih piksela
    new_white_pixels = np.argwhere(new_binary_array == 1)

    # Računamo centar prosjekom
    new_center_x = np.mean(new_white_pixels[:, 1])
    new_center_y = np.mean(new_white_pixels[:, 0])
    new_center = (new_center_x, new_center_y)

    # Računamo radijuse od centra do svakog bijelog piksela
    new_radii = np.sqrt((new_white_pixels[:, 1] - new_center_x) ** 2 + (new_white_pixels[:, 0] - new_center_y) ** 2)

    # Radijus uzimamo kao srednju vrijednost
    new_mean_radius = np.mean(new_radii)

    # Kreiramo novu sliku gdje ćemo imati edge-detection sliku i nacrtat ćemo "fitanu kružnicu" na njoj
    new_output_image = np.copy(new_binary_array)
    new_output_image = new_output_image * 255

    # Crtanje plave kružnice na output slici
    fig, ax = plt.subplots()
    ax.imshow(new_output_image, cmap='gray')

    new_circle = plt.Circle((new_center_x, new_center_y), new_mean_radius, color='cyan', fill=False)
    ax.add_patch(new_circle)

    plt.axis('off')

    # Sačuvamo tu novu sliku
    new_output_path = "edges_with_circle_new.png"
    plt.savefig(new_output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    #KLASIFIKACIJA POMOĆU MSE:

    # Računamo razlike udaljenosti bijelih piksela od izračunatog centra:

    distances = np.sqrt((new_white_pixels[:, 1] - new_center_x) ** 2 + (new_white_pixels[:, 0] - new_center_y) ** 2)
    differences = distances - new_mean_radius

    # Računamo MSE loss funkciju
    mse = np.mean(differences ** 2)

    # Definiramo threshold
    threshold = 50.0

    # Usporedba MSE i thresholda radi klasifikacije
    is_circle = mse < threshold

    print(mse, is_circle)
    return new_output_path, mse, is_circle

if __name__ == '__main__':
    IP = "169.254.105.166"  # Spajanje na IP adresu robota
    PORT = 9559

    if len(sys.argv) > 1:
        IP = sys.argv[1]

    # Redom pozivanje funkcija:

    image_path = showNaoImage(IP, PORT)
    
    edges_path = edge_detection(image_path)

    output_path, mse, is_circle = process_image_and_check_circle(edges_path)

    print(mse)
    print(is_circle)

    # Govor u ovisnosti o klasifikaciji

    tts = ALProxy("ALTextToSpeech", IP, PORT)
    if is_circle:
        tts.say("This is a ball!")
    else:
        tts.say("This is not a ball!")

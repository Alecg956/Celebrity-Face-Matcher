import os
from PIL import Image
import numpy as np
from skimage import color

def load_images(path):
    images = []
    for file in os.listdir(path):
        print(file)
        img = np.array(Image.open((os.path.join(path, file))), dtype=np.float64)
        #TODO resize if needed
        img = color.rgb2gray(img) 
        img = (img - np.mean(img)) / np.std(img)
        images.append(img)
    print(images)
    return images


def main():
    path = ""
    images = load_images(path)
    

if __name__ == "__main__":
    main()

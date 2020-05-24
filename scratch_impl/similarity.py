import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras_vggface.vggface import VGGFace

def preprocess_image(image_path):
    
    # resizes image and changes to keras format
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# code based on https://machinelearningmastery.com/how-to-perform-face
# -recognition-with-vggface2-convolutional-neural-network-in-keras/

def find_cosine_similarity(input_img, celeb_img):
    
    # compute cosine similarity
    a = np.matmul(np.transpose(input_img), celeb_img)
    b = np.sum(np.multiply(input_img, input_img))
    c = np.sum(np.multiply(celeb_img, celeb_img))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def run_similarity_detector(input_source):
    
    # run prediction on input image
    input_img = os.getcwd() + input_source
    input_img = preprocess_image(input_img)
    
    # load resnet, target input shape is 224x224 from preprocessing
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    bestMatch = np.inf
    bestFile = ""
    
    # compute model score for input image
    input_pred = model.predict(input_img)[0,:]
    
    # loop through each celebrity
    for file in os.listdir(os.getcwd() + "/data/celebs/"):
        
        # load celeb image, compute model score
        celeb_img = os.getcwd() + "/data/celebs/" + file
        celeb_pred = model.predict(preprocess_image(celeb_img))[0,:]
        
        # compute similarity between each celeb and input image scores
        sim = find_cosine_similarity(input_pred, celeb_pred)
        
        # update best if similarity is closer
        if sim < bestMatch:
            bestMatch = sim
            bestFile = file
    
    # output best celebrity
    print("your celebrity look-alike is ", bestFile, "\n")

    return bestFile


def plot_celeb_match(input_source, celeb_file):
    
    f = plt.figure()
    f.add_subplot(1,2, 1)

    # show input image
    plt.imshow(image.load_img(os.getcwd() + input_source))
    plt.xticks([]); plt.yticks([])
    f.add_subplot(1,2, 2)

    # show celebrity match
    plt.imshow(image.load_img(os.getcwd() + "/data/celebs/" + celeb_file))
    plt.xticks([]); plt.yticks([])

    plt.savefig(os.getcwd() + "/data/matching_output/face_match_out.png")
    plt.show(block=True)

import os
import sys
import cv2
from utility import *
from cascade import *
from haar_features import *
from similarity import *

def main():
    
    n = len(sys.argv) 

    if n != 2:
        print("\nusage: python3 main.py path/to/image\n")
        return
    
    # load the cascade model from disk
    filename = 'sav_files/cascade_lfw_cbcl_t2_3_features_v1.sav'
    test_cascade = load_from_disk(filename)
    
    # load the image and define the window width and height
    image = load_and_preprocess_image(sys.argv[1])
    
    bounding_boxes = pyramid_detect(image, cascade=test_cascade, cascade_window_size = 24, scale=1.6)

    clone = image.copy()
    cropped = apply_crop(bounding_boxes, clone)
    
    celeb_file = run_similarity_detector("/data/uploads/cropped.jpg")
    plot_celeb_match("/" + sys.argv[1], celeb_file)
    

if __name__ == "__main__":
    main()

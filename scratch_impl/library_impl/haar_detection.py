import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt


def main():
    
    if len(sys.argv) != 2:
        print("usage: python3 haar_detection.py image_name")
        sys.exit(0)
    
    # parse image name
    image_name = str(sys.argv[1])
    
    print("\nrunning haar-cascade detector on " + image_name + "\n")
    
    # load pre-trained face classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # read in input image, convert to grayscale
    img = cv2.imread("input_faces/" + image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    # Parameters:
    # 1. Image
    # 2. scaleFactor : Parameter specifying how much the image size is reduced at each image scale (image pyramid)
    # 3. Parameter specifying how many neighbors each candidate rectangle should have to retain it
    #
    # Returns: a rect with the locations of the faces detected
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # mark the rectangles on the image
    for (x,y,w,h) in faces:
        
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        
    plt.imsave("output_detections/" + "detected_" + image_name,img)

if __name__ == '__main__':
    main()
    
    

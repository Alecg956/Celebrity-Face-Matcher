import numpy as np
import cv2
import pickle
import time
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.feature import haar_like_feature, haar_like_feature_coord
from haar_features import RectangularRegion, HaarFeature
from typing import List
from dask import delayed


def integral_image(input_image: np.ndarray) -> np.ndarray:
    """
    Compute an integral image. Useful for computing Haar-like Features.

    (https://en.wikipedia.org/wiki/Summed-area_table)
    """
    return input_image.cumsum(0).cumsum(1)


def build_features(img_shape) -> List[HaarFeature]:
    """
    Build all possible HaarFeature's for a given image shape.

    Loops over every row and column in the image as well as every possible
    width and height of a RectangularRegion. Appends a 2, 3, or 4 rectangle
    HaarFeature if the given dimensions & position are inside the bounds of
    the image.
    """
    features = []
    
    # iterating over rectangular region height
    for h in range(1, img_shape[0] + 1):
        
        # iterating over rectangular region width
        for w in range(1, img_shape[1] + 1):
            
            # iterating over every valid row with this region width
            for i in range(0, img_shape[0] - h + 1):
                
                # iterating over every valid column with this region height
                for j in range(0, img_shape[1] - w + 1):
                    
                    # immediately available rectangular region
                    current = RectangularRegion(j, i, h, w)
                    
                    # each of these checks if we can fit another region to
                    # the right, below, etc.

                    # horizontal two rectangle feature
                    if j + 2 * w <= img_shape[1]:
                        right = RectangularRegion(j + w, i, h, w)
                        features.append(HaarFeature([right], [current]))

                    # vertical two rectangle feature
                    if i + 2 * h <= img_shape[0]:
                        bottom = RectangularRegion(j, i + h, h, w)
                        features.append(HaarFeature([current], [bottom]))

                    # vertical three rectangle feature
                    if i + 3 * h <= img_shape[0]:
                        features.append(HaarFeature([bottom], [current, RectangularRegion(j, i + 2*h, h, w)]))

                    # horizontal three rectangle feature
                    if j + 3 * w <= img_shape[1]:
                        features.append(HaarFeature([right], [current, RectangularRegion(j + 2*w, i, h, w)]))

                    # four rectangle feature
                    if j + 2 * w <= img_shape[1] and i + 2 * h <= img_shape[0]:
                        features.append(HaarFeature([right, bottom], [current, RectangularRegion(j + w, i + h, h, w)]))
    return features


def apply_features_to_dataset(features: List[HaarFeature],
                              dataset: np.ndarray) -> np.ndarray:
    
    """Given a list of HaarFeature's, computes their value on each image in
    a dataset.

    :param features: array of all possible HaarFeature's for an image of
    shape WxH
    :param dataset: array of size (len_dataset, W, H) consisting of images
    each with dimension WxH

    Returns an array whose (i, j)th element is the result of evaluating
    feature j on image i in dataset.
    """
    
    X = np.empty((len(dataset), len(features)))
    
    # apply integral_image to every image in a fast vectorized way
    integral_images = integral_image(dataset.T).T

    # iterate over every feature
    for i, image in enumerate(dataset):
        
        # and over every image
        for j, f in enumerate(features):
            
            # saving the evaluation of feature i on image j to X[i, j]
            X[i, j] = f.evaluate(integral_images[i])

    return X


@delayed
def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    
    # compute integral image
    ii = integral_image(img)
    
    # return list of all haar features in image
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)


def compute_haar_features(feature_types, images):

    # code based on https://scikit-image.org/docs/dev/auto_examples/
    # applications/plot_haar_extraction_selection_classification.html

    # random seed for repeatability
    np.random.seed(4)

    # Build a computation graph using Dask. This allows the use of multiple
    # CPU cores later during the actual computation
    print("\n building computational graph\n")
    X = delayed(extract_feature_image(img, feature_types) for img in images)

    # Compute the result
    t_start = time()
    print("\ncomputing features\n")
    X = np.array(X.compute(scheduler='threads'))
    time_full_feature_comp = time() - t_start

    print("\ncomputation took: ", time_full_feature_comp, "\n")
    
    # Extract all possible feature coords and feature types
    feature_coord, feature_type = haar_like_feature_coord(width=images.shape[2], height=images.shape[1],
                                  feature_type=feature_types)

    return X, feature_coord, feature_type


# calculate the false positive and negative rate given predictions and labels
def calc_false_preds(pred_y, y):

    false_positives = np.sum((pred_y == 1) & (y == -1))
    false_negatives = np.sum((pred_y == -1) & (y == 1))

    return (false_positives*100)/len(y), (false_negatives*100)/len(y)


# Save object to disk
def save_to_disk(object, filename):
    pickle.dump(object, open(filename, 'wb'))


# load object
def load_from_disk(filename):
    object = pickle.load(open(filename, 'rb'))
    return object

def load_and_preprocess_image(filename):
    
    # load the image and define the window width and height
    image = cv2.imread(filename)
    
    if image is None:
        print("\ncouldn't read specified image file\n")
        return
    
    # preprocess image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.shape[0] > 256 and image.shape[1] > 256:
        
        print("\nresizing image\n")
        image = cv2.resize(image, (256, 256))
        
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    return image


# code based on https://www.pyimagesearch.com/2015/03/23/sliding-windows-
# for-object-detection-with-python-and-opencv/

def sliding_window(image, step_size, window_size):
    
    # slide a window across the image
    
    for y in range(0, image.shape[0] - window_size + 1, step_size):
        
        for x in range(0, image.shape[1] - window_size + 1, step_size):
            
            # yield the current window
            yield (x, y, image[y:y + window_size, x:x + window_size])

            
def pyramid_detect(image, cascade, cascade_window_size, scale=1.3):
    
    # all bounding boxes found organized by window size
    all_rects = []
    
    # initial window size to iterate
    window_size = cascade_window_size
    
    # loop over the image pyramid
    while window_size < np.minimum(image.shape[0], image.shape[1]):
        
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(image, step_size=10, window_size=window_size):
            
            # resize all subwindows to detector size
            window = cv2.resize(window, (cascade_window_size, cascade_window_size))
            
            # Call the cascade function here
            res = cascade.apply_cascade(window)
            
            if res == True:
                
                # record positives
                all_rects.append(((x, y), (x + window_size, y + window_size)))
                
            # draw each subwindow evaluated
            clone = image.copy()
            cv2.rectangle(clone, (x, y), (x + window_size, y + window_size), (0, 255, 255), 2)
            cv2.imshow("Cascade", clone)
            cv2.waitKey(1)
            time.sleep(0.0001)
        
        # increase window size by scale factor
        window_size = int(window_size*scale)
                     
    cv2.destroyWindow("Cascade")
            
    return all_rects


def intersects(self, other):
    
    '''returns whether two rectangles intersect or not'''
    
    # If the rectangles do not intersect, then at least one of the right sides will be to the left 
    # of the left side of the other rectangle (i.e. it will be a separating axis), or vice versa
    trx1 = self[1][0] #top right
    try1 = self[0][1]
    blx1 = self[0][0] #bottom left
    bly1 = self[1][1]
    trx2 = other[1][0] #top right
    try2 = other[0][1]
    blx2 = other[0][0] #bottom left 
    bly2 = other[1][1]

    return not (trx1 < blx2 or blx1 > trx2 or try1 > bly2 or bly1 < try2)


def apply_crop(rect_set, clone):
    
    '''takes a given set of rectangles and determines final bounding box'''
    
    pt1, pt2 = final_output(rect_set)
    cv2.rectangle(clone, pt1, pt2, (0, 255, 255), 2)
    
    crop = clone[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    plt.imsave("data/uploads/cropped.jpg", crop, cmap=cm.gray)
    
    return crop
    

def final_output(rect_set):
    
    '''outputs the average of the intersection of all rectangles in set that has the greatest # of ROIs'''
    sets = []
    
    initial_set = [rect_set[0]]
    sets.append(initial_set)

    # finds all intersecting rectangles, seperates into sets
    for curr_rect in rect_set[1:]:
        
        for setO in sets:
            
            for set_rect in setO:
                
                if intersects(curr_rect, set_rect) and curr_rect is not set_rect:
                    print("intersects")
                    setO.append(curr_rect)

    most_set = sets[np.argmax([len(set) for set in sets])]
    print(most_set)
    
    # finds average of largest set
    xsumtop = 0
    ysumtop = 0
    xsumbot = 0
    ysumbot = 0
    
    for point in most_set:
        xsumtop += point[0][0]
        ysumtop += point[0][1]
        xsumbot += point[1][0]
        ysumbot += point[1][1]

    xsumtop /= len(most_set)
    ysumtop /= len(most_set)
    xsumbot /= len(most_set)
    ysumbot /= len(most_set)

    rect1 = (int(xsumtop), int(ysumtop))
    rect2 = (int(xsumbot), int(ysumbot))

    return rect1, rect2
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cascade import *
from utility import *
from similarity import *
from skimage.data import lfw_subset
from sklearn.datasets import fetch_lfw_people 
from sklearn.model_selection import train_test_split
import cv2
import time
import os
import matplotlib.pyplot as plt
import glob


def test_haar_features():
    
    print("\nloading lfw subset\n")
    images = lfw_subset()
    
    feature_types = ['type-2-x', 'type-2-y']
    
    X, feature_coords, feature_types = compute_haar_features(feature_types, images)
    
    save_to_disk([X, feature_coords, feature_types], "sav_files/lfw_subset_t2_features.sav")
    
    
def test_boost_w_preloaded_haar():
    
    print("\nloading precomputed haar features\n")
    
    # load the computed features from disk
    filename = 'sav_files/lfw_subset_t2_features.sav'
    feature_data = load_from_disk(filename)
    
    X = feature_data[0]
    feature_coords = feature_data[1]
    feature_types = feature_data[2]
    
    # 100 positive and 100 negative images
    num_pos_neg = 100
    
    # Label images (100 faces and 100 non-faces)
    y = np.array([1] * num_pos_neg + [-1] * num_pos_neg)
    
    # X is an array of size (Number of images, Number of Fweatures)
    # y is an array of size (Number of Images, )
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
    
    print("\ntraining ensemble\n")
    
    # run adaboost from scratch
    ensemble = boost_ensemble_scratch()
    ensemble.train(train_X, train_y, feature_coords, feature_types, M=4)
    
    # save the model to disk
    filename = 'sav_files/model_lfw_subset_t2_features.sav'
    save_to_disk(ensemble, filename)
    
    # measure scratch classifier test accuracy (full dataset)
    scratch_preds = ensemble.predict(test_X)
    scratch_test_score = (scratch_preds == test_y).sum() / len(test_y)
    print('\n Adaboost Scratch Test Accuracy = ', scratch_test_score, '\n') 
    
    # compute false negative and positive rate
    false_positives, false_negatives = calc_false_preds(scratch_preds, test_y)

    print("\n false positive %: ", false_positives)
    print("\n false negative %: ", false_negatives)
    
    # reduce false negative rate to 1%
    ensemble.configure_false_neg_percent(test_X, test_y, 1)

    scratch_preds = ensemble.predict(test_X)
    scratch_test_score = (scratch_preds == test_y).sum() / len(test_y)
    print('\n Adaboost Scratch Test Accuracy = ', scratch_test_score, '\n') 
    
    # verify false negative rate was reduced
    false_positives, false_negatives = calc_false_preds(scratch_preds, test_y)

    print("\n false positive %: ", false_positives)
    print("\n false negative %: ", false_negatives)

    
def test_generate_cascade_lfw_subset():
    
    # load the computed features from disk
    filename = 'sav_files/lfw_subset_t2_features.sav'
    feature_data = load_from_disk(filename)
    
    X = feature_data[0]
    feature_coords = feature_data[1]
    feature_types = feature_data[2]
    
    # 100 positive and 100 negative images
    num_pos_neg = 100
    
    # Label images (100 faces and 100 non-faces)
    y = np.array([1] * num_pos_neg + [-1] * num_pos_neg)
    
    # X is an array of size (Number of images, Number of Features)
    # y is an array of size (Number of Images, )
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
    
    test_cascade = cascade()
    test_cascade.train(train_X, train_y, feature_coords, feature_types, num_stages = 5)
    
    print("\ncascade:\n")
    print(test_cascade.ensembles)
    
    # save the model to disk
    filename = 'sav_files/cascade_lfw_subset_t2_features.sav'
    save_to_disk(test_cascade, filename)

    
    
def test_apply_cascade_lfw_full():
    
    # note: this tests on the novel faces, but we still expect good results (currently around ~90%)
    
    # load the cascade model from disk
    filename = 'sav_files/cascade_lfw_cbcl_t2_3_features_v1.sav'
    cascade = load_from_disk(filename)
    
    # load in lfw_subset
    dataset = fetch_lfw_people()
    subwindows = dataset.images[500:1500]
    
    
    image_num = 0
    num_correct = 0
    
    print("\ntesting positive lfw images\n")
    
    for window in subwindows:
        
        window = cv2.resize(window, (24, 24))
        
        # should pass the cascade since it is positive
        res = cascade.apply_cascade(window)
    
        # we expect these images to be true because they are faces
        if(res != True):
            
            print("\nimage num: ", image_num, " failed\n")
            
        else:
            num_correct += 1
            
        image_num += 1
    
    print("\ncorrectly predicted images: ", num_correct)
    print("\ntotal images: ", subwindows.shape[0], "\n")
    print("\npercent correct: ", num_correct/subwindows.shape[0], "\n")
    
    
def test_apply_cascade_non_faces():
    
    # note: this tests on the novel faces, but we still expect good results (currently ~85%)
    
    # load the cascade model from disk
    filename = 'sav_files/cascade_lfw_cbcl_t2_3_features_v1.sav'
    cascade = load_from_disk(filename)
    
    image_num = 0
    num_correct = 0
    
    print("\ntesting negative lfw images\n")
    
    for f in glob.iglob("data/test/non-face/*"):
        
        window = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        window = cv2.resize(window, dsize=(24, 24))
        window = (window - np.min(window)) / (np.max(window) - np.min(window))

        
        # should pass the cascade since it is positive
        res = cascade.apply_cascade(window)
    
        # we expect these images to be true because they are faces
        if(res != False):
            
            print("\nimage num: ", image_num, " failed\n")
            
        else:
            num_correct += 1
            
        image_num += 1
    
    print("\ncorrectly predicted images: ", num_correct)
    print("\ntotal images: ", image_num, "\n")
    print("\npercent correct: ", num_correct/image_num, "\n")
    
    
def test_sliding_window_with_pyramid():
    
# load the cascade model from disk
    filename = 'sav_files/cascade_lfw_cbcl_t2_3_features_v1.sav'
    test_cascade = load_from_disk(filename)
    
    # load the image and define the window width and height
    image = cv2.imread("data/test/pyramid/pyramid_test2.jpg")
    
    # preprocess image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    bounding_boxes = pyramid_detect(image, cascade=test_cascade, cascade_window_size = 24, scale=1.4)
    
    ROI_num = 0
    for rect_set in bounding_boxes:
        
        clone = image.copy()
        
        for rect in rect_set:
            cv2.rectangle(clone, rect[0], rect[1], (0, 255, 255), 2)

        cv2.imshow("ROIs " + str(ROI_num), clone)
        
        ROI_num += 1
        
    cv2.waitKey(0)

def test_sliding_window_with_pyramid_and_similarity():
    
    # load the cascade model from disk
    filename = 'sav_files/cascade_lfw_cbcl_t2_3_features_v1.sav'
    test_cascade = load_from_disk(filename)
    
    # load the image and define the window width and height
    image = load_and_preprocess_image("data/test/pyramid/pyramid_test2.jpg")
    
    bounding_boxes = pyramid_detect(image, cascade=test_cascade, cascade_window_size = 24, scale=1.6)

    clone = image.copy()
    cropped = apply_crop(bounding_boxes, clone)
    
    celeb_file = run_similarity_detector("/data/uploads/cropped.jpg")
    plot_celeb_match("/data/test/pyramid/pyramid_test2.jpg", celeb_file)

    
def main():
    
    # Note: each test relies on the previous tests
    
    # test generates the haar features for lfw_subset and saves
#     test_haar_features()

    # test generates a classifier based on preloaded lfw_subset t2 features
#     test_boost_w_preloaded_haar()
    
    # test trains a cascade of adaboost ensembles
#     test_generate_cascade_lfw_subset()
    
    # test a trained cascade on lfw full dataset
#     test_apply_cascade_lfw_full()

    # test a trained cascade on non-faces from cbcl dataset
#     test_apply_cascade_non_faces()

    # test pyramid sliding window for face detection
   #test_sliding_window_with_pyramid() 

    test_sliding_window_with_pyramid_and_similarity()
    


if __name__ == "__main__":
    main()
    
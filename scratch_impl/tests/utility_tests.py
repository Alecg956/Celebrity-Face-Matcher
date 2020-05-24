import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utility import integral_image, build_features, apply_features_to_dataset
from skimage.feature import haar_like_feature


def test_integral_image():
    arr = np.arange(1, 17).reshape((4, 4))
    """
    arr = 
        1  2  3  4z
        5  6  7  8
        9  10 11 12
        13 14 15 16
    """
    ii = integral_image(arr)
    
    # last value should be equal to the sum of all elements in the array
    assert ii[-1, -1] == np.sum(arr)
    
    # first value should be equal to the first element in the array
    assert ii[0, 0] == arr[0, 0]
    
    # last value in first row should be the sum of the row
    assert ii[0, -1] == np.sum(arr[0])
    
    # last value in the first col should be the sum of the col
    assert ii[-1, 0] == np.sum(arr[:, 0])
    
    # should be equal to the sum of everything in arr but the last row & col
    assert ii[2, 2] == np.sum(arr[0:-1, 0:-1])


def test_build_features():
    
    width = 24
    shape = (width, width)
    features = build_features(shape)
    
    print('Expected number of features = 162336')
    print(f'Got number of features = {len(features)}')
    assert len(features) == 162336


def scikit_feature(img, feature):
    
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, img.shape[0], img.shape[1],
                             feature_type=feature)


def test_apply_features_to_dataset():
    
    np.random.seed(0)
    # size of image on each side
    
    width = 3
    height = 3
    num_images = 1
    
    # creating 16 random "images" to test on
    test_arrays = \
        np.random.rand(width * height * num_images).reshape((num_images,
                                                            height,
                                                            width))

    # building and applying features
    features = build_features(test_arrays[0].shape)
    applied_features = apply_features_to_dataset(features, test_arrays)

    # need small epsilon to compare difference between floats
    epsilon = 10e-5

    # now for each "image", test if manually applying every feature to it is
    # equal to what is output from apply_features_to_dataset
    for index, image in enumerate(test_arrays):
        ii = integral_image(image)
        manually_applied_features = np.array([f.evaluate(ii) for f in
                                              features])
        assert ((applied_features[index] - manually_applied_features) <
                epsilon).all()

    # feature_types = ['type-2-x']
    #     # ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']
    # scikit_features = np.vstack(
    #     (np.array(scikit_feature(img, feature_types)) for img in test_arrays))
    # fig, ax = plt.subplots(1, 1)
    # coord, _ = haar_like_feature_coord(height, width, 'type-2-x')
    # print(coord)


def run_all_tests():
    """Run all tests here."""
    test_integral_image()
    test_build_features()
    test_apply_features_to_dataset()


if __name__ == '__main__':
    run_all_tests()

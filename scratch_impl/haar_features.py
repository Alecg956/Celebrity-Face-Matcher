import numpy as np
from typing import List

# code based on https://towardsdatascience.com/computer-vision-detecting-objects-using-haar-cascade-classifier-4585472829a9

class RectangularRegion:
    """
    Represents a rectangular region of an image. Multiple make up a Haar
    Feature.
    """

    def __init__(self, x: int, y: int, height: int, width: int):
        """
        :param int x: x coordinate of this region's upper left corner.
        :param int y: y coordinate of this region's upper left corner.
        :param int height: Height of region, including the upper left pixel.
        :param int width: Width of the region, including the upper left pixel.
        """
        self.x, self.y, self.height, self.width = x, y, height, width

    def evaluate(self, ii: np.ndarray):
        """Sum the region.

        :param np.ndarray ii: ii is an integral_image and the summation of
        this region is with respect to the image that ii is calculated from.
        """
        
        val = ii[self.y][self.x] + \
              ii[self.y + self.height][self.x + self.width] - \
              ii[self.y + self.height][self.x] - \
              ii[self.y][self.x + self.width]

        return val
    
    def __repr__(self):
        """Create basic string representation."""
        return f'Region: x={self.x}, y={self.y}, w={self.width}, ' \
               f'h={self.height}'


class HaarFeature:
    """
    Represents a HaarFeature.

    Consists of a tuple of 'positive' RectangularRegion's and of a tuple of
    'negative' RectangularRegion's.

    The evaluation of this feature is simply the sum of evaluating all
    positive RectangularRegion's from all the negative.
    """

    def __init__(self, positive_regions: List[RectangularRegion],
                 negative_regions: List[RectangularRegion]):
        """
        :param positive_regions: List of RectangularRegion's that contribute
        positively to this feature's evaluation.
        :param negative_regions: List of RectangularRegion's that contribute
        negatively to this feature's evaluation.
        """
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions

    def evaluate(self, ii: np.ndarray):
        """
        Evaluate the HaarFeature.

        :param np.ndarray ii: ii is an integral_image and the evaluation of
        this region is with respect to the image that ii is calculated from.
        """
        
        # needed to ensure proper evaluation of area including edges
        ii = np.pad(ii, 1)
        sum_pos_regions = np.sum([rec.evaluate(ii) for rec in self.positive_regions])
        sum_neg_regions = np.sum([rec.evaluate(ii) for rec in self.negative_regions])

        return sum_pos_regions - sum_neg_regions
               

    def __repr__(self):
        
        """Create basic string representation."""
        
        return f'HaarFeature:\n' + \
               '\t'.join(str(region) + '\n' for region in
                         (self.positive_regions + self.negative_regions))

    
def extract_haar_features(feature_coords):
        
        features = []
        
        for coord in feature_coords:
            
            region_a = []
            region_b = []

            # coord will be an iterable with length equal to the number of
            # rectangles in the HaarFeature it represents
            
            for i in range(len(coord)):
                
                # each rectangle in coord alternates in its HaarFeature sign
                if i % 2 == 0:
                    l = region_a
                else:
                    l = region_b
                    
                # grab x, y, height, width of coord
                # height calculated as difference of y coords + 1
                # width calculated as difference of x coords + 1
                l.append(RectangularRegion(coord[i][0][1],
                                           coord[i][0][0],
                                           coord[i][1][0] - coord[i][0][0] + 1,
                                           coord[i][1][1] - coord[i][0][1] + 1)
                         )

            features.append(HaarFeature(region_b, region_a))

        return np.array(features)
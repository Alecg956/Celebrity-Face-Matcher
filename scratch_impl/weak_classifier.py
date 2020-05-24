import numpy as np


class weak_classifier_scratch:

    # Code based on http://cs229.stanford.edu/extra-notes/boosting.pdf
    
    def __init__(self):
        
        self.threshold = None
        self.prediction = []
        self.minError = np.inf
        self.polarity = 1
        self.feature = -1

    # Build decision stump
    def build_decisionStump(self, X, y, weights):
        
        # Store the number of images and features
        m, n = np.shape(X)

        # Loop through all features
        for feature in range(n):
            
            # Find all of the the unique values in X for each feature
            feature_values = np.expand_dims(X[:, feature], axis=1)
            unique_values = np.unique(feature_values)

            # Loop through unique values
            for threshold in unique_values:
                p = 1

                # Predict the label for each unique value
                prediction = np.ones(np.shape(y), dtype = int)
                prediction[X[:, feature] > threshold] = -1

                # Find the weighted error for the prediction
                error = sum(weights[y != prediction])
                
                # If the error is less than 0.5, invert the error and polarity
                if error > 0.5:
                    error = 1 - error
                    p = -1
                
                if error < self.minError:
                    
                    print(error)
                    # Set the values of min error and the best stumpf
                    self.minError = error
                    self.prediction = prediction.copy()
                    self.threshold = threshold
                    self.polarity = p
                    self.feature = feature


    def predict(self, X):
    
        # Store the number of predicionts
        predictions = np.ones(np.shape(X)[0], dtype = int)

        # Find the non-face id's
        negative_idx = (X[:, self.feature] >= self.threshold)

        # Set non-face id's to -1
        predictions[negative_idx] = -1

        return predictions

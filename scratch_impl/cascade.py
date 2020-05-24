from ensemble import *
from haar_features import extract_haar_features
from utility import *


class cascade:

    def __init__(self):
        self.ensembles = []
        self.feature_count = [1, 5, 10, 25, 50, 50, 100, 150, 200, 250, 500, 500, 750, 1000, 1000, 1500, 1750, 2000, 4000, 6000]

    def train(self, X,y, feature_coords, feature_types, num_stages = 10):
        
        print("\n training cascade with ", num_stages, " stages\n")
        
        # one ensemble per stage
        for i in range(num_stages):
            
            print("\nstage ", i, "\n")
            
            # compute ensembles with # of features and desired false negative percent
            ensemble = boost_ensemble_scratch()
            ensemble.train(X, y, feature_coords, feature_types, M = self.feature_count[i])
            ensemble.configure_false_neg_percent(X, y, 1.0)
            self.ensembles.append(ensemble)
            
    def apply_cascade(self, subwindow):
        
        stage = 0
        
        # apply each ensemble, only continue if passes
        for ensemble in self.ensembles:
            
            # 1. extract haar features needed by current stage
            haar_features = extract_haar_features(ensemble.feature_coords)
            ii = integral_image(subwindow)
            
            evaluated_features = []
            
            # collect evaluated haar features
            for f in haar_features:
        
                evaluated_features.append(f.evaluate(ii))
            
            # 2. place extracted features at correct spot in input array
            X = np.zeros(ensemble.input_length)
            
            # set the feature_indices to the computed features, rest = 0
            X[ensemble.feature_indices] = evaluated_features
            X = X.reshape((1, -1))
    
            # 3. apply current ensemble to the extracted features
            out = ensemble.predict(X)
            
            # 4. stop early if stage says not a face
            if out[0] == -1:
                
                return False
           
            stage += 1
        
        # only return true if input passes all stages
        return True

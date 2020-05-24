from sklearn.tree import DecisionTreeClassifier
from utility import *

class boost_ensemble_scratch:

    def __init__(self):
        
        # each weak classifier
        self.ensemble = []
        
        # weights of each weak classifier
        self.ensemble_weights = []
        
        # number of columns in the input data
        self.input_length = 0
        
        # index in input matrix that feature comes from for each weak classifier
        self.feature_indices = []
        
        # coords of feature for each weak classifier
        self.feature_coords = []
        
        # type of feature for each weak classifier
        self.feature_types = []
        
        # final decision threshold
        self.threshold = 0.5

        
    def train(self, X,y, feature_coords, feature_types, M=10, learning_rate = 1.0):

        ''' Construct an ensemble of weak classifiers from the training data '''

        # code based on https://xavierbourretsicotte.github.io/AdaBoost.html
        # and https://towardsdatascience.com/machine-learning-part-17-boosting-
        # algorithms-adaboost-in-python-d00faac6c464
        
        # record input length for later usage
        self.input_length = X.shape[1]
        
        # store each weak classifier
        classifier_list = []

        # store predictions from each weak classifier per iteration for training accuracy metric
        y_predict_list = []

        # store weight given to each weak classifier in final ensemble per iteration
        classifier_weight_list = []

        # Initialize the sample weights to be all equal
        sample_weight = np.ones(len(y)) / len(y)

        # the number of loops specifies the number of weak classifiers in the final ensemble
        for m in range(M):

            # Fit a weak classifier, depth = 1 and lead_nodes = 2 ensures it's a stump
            classifier = DecisionTreeClassifier(max_depth = 1, max_leaf_nodes=2)
            classifier.fit(X, y, sample_weight=sample_weight)
            
            # record feature index, coords, and type this weak classifier is based on     
            self.feature_indices.append(np.argmax(classifier.feature_importances_))
            self.feature_coords.append(feature_coords[self.feature_indices[m]])
            self.feature_types.append(feature_types[self.feature_indices[m]])
            
            # generate the predictions based on the weak classifier
            y_predict = classifier.predict(X)

            # identify misclassifications
            incorrect = (y_predict != y)

            # classifier error, sample-weighted average over incorrect predictions
            classifier_error = np.average(incorrect, weights=sample_weight, axis=0)

            # this classifier's weight = learning rate * log(1-error / error)
            classifier_weight =  learning_rate * 0.5 * np.log((1. - classifier_error) / classifier_error)

            # weight decreases if y == y_predict bc already correct, exp = negative
            # increases if y != y_predict bc needs more priority on the next iteration, exp = positive
            sample_weight *= np.exp(-1. * classifier_weight * y * y_predict)

            # ensure all sample weights add to 1
            sample_weight /= np.sum(sample_weight)

            # Save iteration values
            classifier_list.append(classifier)
            y_predict_list.append(y_predict.copy())
            classifier_weight_list.append(classifier_weight.copy())


        classifier_list = np.asarray(classifier_list)
        y_predict_list = np.asarray(y_predict_list)
        classifier_weight_list = np.reshape(np.asarray(classifier_weight_list), (M,1))

        # weight of each classifier * the prediction of that classifier for each output
        weighted_preds = y_predict_list * classifier_weight_list

        # make final predictions: sign turns any negative sums to -1 and positive to 1
        preds = np.array(np.sign(np.sum(weighted_preds, axis=0)))
        print('\n Adaboost Scratch Train Accuracy = ', (preds == y).sum() / len(y), '\n') 

        self.ensemble = np.array(classifier_list)
        self.ensemble_weights = np.array(classifier_weight_list)


    def configure_false_neg_percent(self, X, y, neg_percent):
        
        ''' Assumes classifier already trained and takes in test data to configure rate on '''

        # initial prediction and measurements
        y_preds = self.predict(X)
        false_positives, false_negatives = calc_false_preds(y_preds, y)

        # keep relaxing the boundary until enough negatives -> positives to satisfy desired rate
        while (false_negatives > neg_percent):

            self.threshold -= 0.05
            y_preds = self.predict(X)
            false_positives, false_negatives = calc_false_preds(y_preds, y)


    def predict(self, X):

        ''' make a prediction based on the supplied weak classifiers '''
        
        # store predictions of each weak classifier
        predictions = []

        for weak_classifier in self.ensemble:

            # generate prediction for X by weak_classifier
            predictions.append(weak_classifier.predict(X).copy())

        # convert -1s to 0 for computation purposes
        for weak_pred in predictions:
            weak_pred[weak_pred==-1] = 0

        # weight each prediction, sum for final result
        weighted_predictions = np.sum(predictions * self.ensemble_weights, axis = 0)

        # this value determines the final decisions
        boundary = (self.threshold*np.sum(self.ensemble_weights))

        # turn weighted predictions into final decisions based on boundary
        final_decisions = np.ones(weighted_predictions.shape)
        final_decisions[weighted_predictions < boundary] = -1

        # converted weighted sum to -1 or 1 for final decision
        return final_decisions

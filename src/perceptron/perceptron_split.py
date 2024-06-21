import sys
import numpy as np
import pandas as pd

class perceptron:
    #initialize perceptron
    def __init__ (self, learningrate, iterations = 100):
        self.learningrate = learningrate
        self.iterations = iterations
        self.step_func = self.step_func
        self.weights = None
        self.highest_accuracy = 0
        self.iters = 0
      #step function
    def step_func(self, features):
        features_array = np.array(features) #convert features -> array
        return np.where(features_array > 0, 1, 0) #1 if features > 0 else 0

    #training function
    def training(self, features, labels):
        num_samples, num_features = features.shape #get shape of features array
        
        #self.weights = np.zeros_like(features[0]) #weights with zeros
        self.weights = np.random.uniform(-1, 1, num_features) #weights with random vals in -1 to 1
        
        total_iters = 0 #total iters performed
        
        while self.iters < self.iterations: #while iters < 100
            is_label = 0 #correct label
            for i in range(num_samples): #iterate through samples
                output = np.dot(self.weights, features[i]) #dotproduct -> weights and features[i]
                prediction = self.step_func(output) #calculate prediction -> step_func(output)
                if prediction == labels[i]: #prediction is curr label
                    is_label += 1        #increment corr label
                error = labels[i] - prediction #calculate error 
                self.update_weights(features[i], error) #update weights -> curr feature and error
            
            accuracy = is_label / num_samples #calculate accuracy
            #FOR TESTING ACCURACY AND WEIGHTS AT EACH ITERATION
            # print("accuracy at iteration", self.iters + 1, "accuracy: ", 
            #       accuracy, "\nweights for iteration: ", self.weights)
            #print("accuracy at iteration", self.iters + 1, "accuracy: ", accuracy)
            if accuracy > self.highest_accuracy:#if accuracy> highest accuracy
                self.highest_accuracy = accuracy #update highest accuracy
                best_weights = self.weights.copy()#update best weights
                self.iters = 0 #reset iters
            self.iters += 1
            total_iters += 1

        #PRINT FINAL RESULTS
        print("Total amount of iterations performed: ", total_iters)
        print("Highest accuracy: ", self.highest_accuracy)
        print("Final weights: ", best_weights)
    
    #update weights function
    def update_weights(self, features, error): #update weights
        self.weights += self.learningrate * error * features #update weights -> learningrate * error * features

    #predict function for test data
    def predict(self, features):
        predictions = [] #store predictions
        for feature in features: #iterate through features
            output = np.dot(self.weights, feature) #dotproduct -> weights and feature
            prediction = self.step_func(output) #calculate prediction -> step_func(output)
            predictions += [prediction]#update predictions
        return predictions, self.weights #return test predictions and test weights
 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("PROPER: python3 perceptron_test.py ionosphere.data")
        sys.exit(1)
    
    data = pd.read_csv(sys.argv[1], header=0, sep=' ') #reading csv data
    features = data.iloc[:, :-1].values #get features from data
    labels = data.iloc[:, -1].values #get labels from data
    features = np.insert(features, 0, 1, axis=1) #insert 1 at index 0 in features (bias)
    
    #SPLIT DATA INTO TRAIN AND TEST
    num_samples = len(features) #get number of samples
    test_size = int(num_samples * 0.2) # 20% test data
    test_indices = np.random.choice(num_samples, test_size, replace=False) #random test indices
    train_indices = np.random.choice(np.arange(num_samples), num_samples - test_size, replace=False) #random train indices
    features_train, features_test = features[train_indices], features[test_indices] #split features for train and test
    labels_train, labels_test = labels[train_indices], labels[test_indices] #split labels for train and test

    labels = np.where(labels == 'g', 1, 0).astype(int)#convert labels to 1 if 'g' else 0
    
    p = perceptron(learningrate=0.01) #initialize perceptron
    p.training(features, labels) #train perceptron
    
    test_predictions, weights = p.predict(features_test)#predict test data
    
    accuracy = np.mean(test_predictions == (labels_test == 'g'))#calculate accuracy of test data
    #PRINT TEST ACCURACY
    print("Test Accuracy:", accuracy)
    print("Test Weights", weights)
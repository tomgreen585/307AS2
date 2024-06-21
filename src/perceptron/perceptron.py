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
        
        self.weights = np.zeros_like(features[0]) #weights at zeros
        #self.weights = np.random.uniform(-1, 1, num_features) #weights with random vals in -1 to 1
        
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
                bw = self.weights.copy()#update best weights
                self.iters = 0 #reset iters
            self.iters += 1
            total_iters += 1

        #PRINT FINAL RESULTS
        print("Total amount of iterations performed: ", total_iters)
        print("Highest accuracy: ", self.highest_accuracy)
        print("Final weights: ", bw)
    
    #update weights function
    def update_weights(self, features, error): #update weights
        self.weights += self.learningrate * error * features #update weights -> learningrate * error * features
 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("PROPER: python3 perceptron_test.py ionosphere.data")
        sys.exit(1)
    
    data = pd.read_csv(sys.argv[1], header=0, sep=' ') #reading csv data
    features = data.iloc[:, :-1].values #get features from data
    labels = data.iloc[:, -1].values #get labels from data
    features = np.insert(features, 0, 1, axis=1) #insert 1 at index 0 in features (bias)
    labels = np.where(labels == 'g', 1, 0).astype(int)#convert labels to 1 if 'g' else 0
    
    p = perceptron(learningrate=0.01) #initialize perceptron
    p.training(features, labels) #train perceptron
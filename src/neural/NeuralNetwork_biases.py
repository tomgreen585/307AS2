import numpy as np

class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, hbiases, obiases, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights
        self.learning_rate = learning_rate
        self.hbiases = hbiases #initialize hidden biases
        self.obiases = obiases #initialize output biases

    # Calculate neuron activation for an input
    def sigmoid(self, input):
        #output = np.NaN
        output = 1 / (1 + np.exp(-input)) #sigmoid calculation
        return output
    
    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        #feedforward
        #for each example z, calculate the network output Oz with current weights
        #calcaulte average loss(objective) function as a function of parameters, J(W), over batch of z
        hidden_layer_outputs = []
        for i in range(self.num_hidden): 
            # TODO! Calculate the weighted sum, and then compute the final output.
            weighted_sum = 0
            for j in range(self.num_inputs): #iterate over num_inputs
                weighted_sum += inputs[j] * self.hidden_layer_weights[j][i] #calculate weightedsum -> inputs x hiddenweights
            weighted_sum += self.hbiases[i] #add bias
            output = self.sigmoid(weighted_sum) #apply weightedsum to sigmoid
            hidden_layer_outputs.append(output) #append output to hidden output

        output_layer_outputs = []
        for i in range(self.num_outputs): 
            # TODO! Calculate the weighted sum, and then compute the final output.
            weighted_sum = 0
            for j in range(self.num_hidden):#iterate over num_hidden
                weighted_sum += hidden_layer_outputs[j] * self.output_layer_weights[j][i]  #calculate weightedsum -> hiddenoutputs x outputweights
            weighted_sum += self.obiases[i] #add bias
            output = self.sigmoid(weighted_sum) #apply weightedsum to sigmoid
            output_layer_outputs.append(output) #append output to output layer

        return hidden_layer_outputs, output_layer_outputs  # Return the hidden layer outputs and output layer outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):

        output_layer_betas = np.zeros(self.num_outputs)
        # Output node k: 𝛽k = 𝑑k(s) − 𝑜k(s) (slope/derivative of loss)
        # Calculate output layer betas.
        #output = desired_outputs - output_layer_outputs
        for k in range(self.num_outputs): #iterate over num_outputs
            output = desired_outputs - output_layer_outputs #use output node calculation -> desired - output
            output_layer_betas[k] = output[k] * output_layer_outputs[k] * (1 - output_layer_outputs[k]) #calculate output layer betas using node calc

        hidden_layer_betas = np.zeros(self.num_hidden)
        #Hidden node j: 𝛽j= ∑k wj→k 𝑜_𝑘 (1 − 𝑜k) 𝛽k
        # Calculate hidden layer betas.
        #print('HL betas: ', hidden_layer_betas)
        error = np.zeros(self.num_hidden)#initialize error array
        for j in range(self.num_hidden): #iterate over num_hidden
            for k in range(self.num_outputs): #iterate over num_outputs
                error[j] += self.output_layer_weights[j][k] * output_layer_betas[k] #calculate error -> outputweights x outputlayerbetas
            hidden_layer_betas[j] += error[j] * hidden_layer_outputs[j] * (1 - hidden_layer_outputs[j])#calculate hidden layer betas using hidden node equation

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        # Calculate output layer weight changes.
        # Compute the weight changes Δ𝑤j→k = − n oj ok (1 − 𝑜k)𝛽k
        for k in range(self.num_outputs): #iterate over num_outputs
            self.obiases[k] += self.learning_rate * output_layer_betas[k] #update bias 
            for j in range(self.num_hidden):   #iterate over num_hidden
                delta_output_layer_weights[j, k] = self.learning_rate * hidden_layer_outputs[j] * output_layer_betas[k] #calculate weight changes using equation

        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        # Compute the weight changes Δ𝑤j→k = − n oj ok (1 − 𝑜k)𝛽k
        # Calculate hidden layer weight changes.
        for k in range(self.num_hidden): #iterate over num_hidden
            self.hbiases[k] += self.learning_rate * hidden_layer_betas[k]       #update bias
            for j in range(self.num_inputs):#iterate over num_inputs
                delta_hidden_layer_weights[j, k] = self.learning_rate * inputs[j] * hidden_layer_betas[k] #calculate weight changes using equation

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights):
        # Update the weights.
        # Update: Δ𝑤j→k ∝ − 𝑜j × 𝑠𝑙𝑜𝑝𝑒k × 𝛽k = −𝑜j 𝑜k (1 − 𝑜k) 𝛽k
        for j in range(self.num_hidden): #iterate over num_hidden
            for k in range(self.num_outputs): #iterate over num_outputs
                self.output_layer_weights[j][k] += delta_output_layer_weights[j][k] #update output layer weights += deltaweights
                 
        for j in range(self.num_inputs):#iterate over num_inputs
            for k in range(self.num_hidden):#iterate over num_hidden
                self.hidden_layer_weights[j][k] += delta_hidden_layer_weights[j][k]#update hidden layer weights += deltaweights

    def train(self, instances, desired_outputs, epochs):

        for epoch in range(epochs):
            print('epoch = ', epoch)
            predictions = []
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])
                #predicted_class = None  # TODO!
                predicted_class = np.argmax(output_layer_outputs) #get index of max output
                predictions.append(predicted_class) #update predictions with predictedclass

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)

            # Print new weights
            # print('Hidden layer weights \n', self.hidden_layer_weights)
            # print('Output layer weights  \n', self.output_layer_weights)

            # TODO: Print accuracy achieved over this epoch
            acc = None
            total = len(instances) #number of instances
            c = 0 #counter
            for i in range(total): #iterate over predictions
                if(desired_outputs[i][predictions[i]] == 1):
                    c += 1 #increment counter
            acc = c/total #calculate accuracy
            print('epoch acc = ', acc, '%') #print

    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            #print(output_layer_outputs)
            #predicted_class = None  # TODO! Should be 0, 1, or 2.
            predicted_class = np.argmax(output_layer_outputs) #get index of max output
            predictions.append(predicted_class) #append to predictions
        return predictions
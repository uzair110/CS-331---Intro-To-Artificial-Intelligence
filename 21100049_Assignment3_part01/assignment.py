# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:00:51 2019

@author: YourAverageSciencePal
"""
import numpy as np
import sys
import time
import sklearn
import pickle
from sklearn.preprocessing import OneHotEncoder
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy import misc
#import matplotlib.pyplot as plt


'''
Depending on your choice of library you have to install that library using pip
'''


'''
Read chapter on neural network from the book. Most of the derivatives,formulas 
are already there.
Before starting this assignment. Familarize yourselves with np.dot(),
What is meant by a*b in 2 numpy arrays.
What is difference between np.matmul and a*b and np.dot.
Numpy already has vectorized functions for addition and subtraction and even for division
For transpose just do a.T where a is a numpy array 
Also search how to call a static method in a class.
If there is some error. You will get error in shapes dimensions not matched
because a*b !=b*a in matrices
'''

class NeuralNetwork():
	@staticmethod
	#note the self argument is missing i.e. why you have to search how to use static methods/functions
	def cross_entropy_loss(y_pred, y_true):
		'''implement cross_entropy loss error function here
		Hint: Numpy has a sum function already
		Numpy has also a log function
		Remember loss is a number so if y_pred and y_true are arrays you have to sum them in the end
		after calculating -[y_true*log(y_pred)]'''
		return -np.sum(y_true*np.log(y_pred))
	@staticmethod
	def accuracy(y_pred, y_true):
		'''function to calculate accuracy of the two lists/arrays
		Accuracy = (number of same elements at same position in both arrays)/total length of any array
		Ex-> y_pred = np.array([1,2,3]) y_true=np.array([1,2,4]) Accuracy = 2/3*100 (2 Matches and 1 Mismatch)'''
		return (np.sum(y_pred == y_true))

	
	@staticmethod
	def softmax(x):
		'''Implement the softmax function using numpy here
		Hint: Numpy sum has a parameter axis to sum across row or column. You have to use that
		Use keepdims=True for broadcasting
		You guys should have a pretty good idea what the size of returned value is.
		'''
		return np.exp(x)/np.sum(np.exp(x), axis=0)
	
	@staticmethod
	def sigmoid(x):
		'''Implement the sigmoid function using numpy here
		Sigmoid function is 1/(1+e^(-x))
		Numpy even has a exp function search for it.Eh?
		'''
		sigma = 1/(1+ np.exp(-x))
		return sigma
	
	def __init__(self):
		'''Creates a Feed-Forward Neural Network.
		"nodes_per_layer" is a list containing number of nodes in each layer (including input layer)
		"num_layers" is the number of layers in your network 
		"input_shape" is the shape of the image you are feeding to the network
		"output_shape" is the number of probabilities you are expecting from your network'''

		self.num_layers = 3 # includes input layer
		self.nodes_per_layer = [784, 30, 10]
		self.input_shape = 784
		self.output_shape = 10
		self.__init_weights(self.nodes_per_layer)

	def __init_weights(self, nodes_per_layer):
		'''Initializes all weights and biases between -1 and 1 using numpy'''
		self.weights_ = []
		self.biases_ = []
		for i,_ in enumerate(nodes_per_layer):
			if i == 0:
				# skip input layer, it does not have weights/bias
				continue
			
			weight_matrix = 2 * np.random.random(size=(nodes_per_layer[i-1], nodes_per_layer[i])) - 1
			self.weights_.append(weight_matrix)
			#print(self.weights_)
			bias_vector = 2 * np.random.uniform(size=(nodes_per_layer[i])) - 1
			self.biases_.append(bias_vector)
		# print(len(self.weights_[0]))
		# print(len(self.weights_[1]))
		# print((self.weights_[1][0]))
		# print((self.weights_[1][1]))


	
	def fit(self, Xs, Ys, epochs, t_array, acc_array, startTime, lr=1e-3):
		'''Trains the model on the given dataset for "epoch" number of itterations with step size="lr". 
		Returns list containing loss for each epoch.'''
		history = []
		first_layer = []
		for i in range(epochs):
			for j in range(0,Xs.shape[0]):
				X = Xs[j,:].reshape((1, self.input_shape))
				Y = Ys[j,:].reshape((1, self.output_shape))
				activations=self.forward_pass(X)
				deltas = self.backward_pass(Y, activations)
				first_layer = [X] + activations
				self.weight_update(deltas, first_layer, lr)
				# loss, acc = self.evaluate(Xs, Ys)
				# acc_array.append(acc/float(60000))
				# t_array.append(time.time()-startTime)
			
			loss, acc = self.evaluate(Xs, Ys)
			history.append(loss)

	   


		return history
	
	
	
	def forward_pass(self, input_data):
		'''Executes the feed forward algorithm.
		"input_data" is the input to the network in row-major form
		Returns "activations", which is a list of all layer outputs (excluding input layer of course)
		What is activation?
		In neural network you have inputs(x) and weights(w).
		What is first layer? It is your input right?
		A linear neuron is this: y = w.T*x+b =>T is the transpose operator 
		A sigmoid neuron activation is y = sigmoid(w1.T*x+b1) for 1st hidden layer 
		Now for the last hidden layer the activation y = sigmoid(w2.T*y+b2).
		'''

		activations = []
		hidden_layer = self.sigmoid(np.dot(input_data, self.weights_[0]) + self.biases_[0])
		# print(hidden_layer.shape[0])
	  
		activations.append(hidden_layer)
		output_layer = self.sigmoid(np.dot(hidden_layer, self.weights_[1])+ self.biases_[1])
		# print(output_layer.shape[0])
		activations.append(output_layer)
		
	  
	  
		return activations
	
	def backward_pass(self, targets, layer_activations):
		'''Executes the backpropogation algorithm.
		"targets" is the ground truth/labels
		"layer_activations" are the return value of the forward pass step
		Returns "deltas", which is a list containing weight update values for all layers (excluding the input layer of course)
		You need to work on the paper to develop a generalized formulae before implementing this.
		Chain rule and derivatives are the pre-requisite for this part.
		'''
		deltas = []

		difference_outer_layer = (layer_activations[1] - targets)
		derivative_outer_layer = layer_activations[1]*(1-layer_activations[1]) #sigmoid
		delta_outer_layer = np.array([difference_outer_layer*derivative_outer_layer])
		a,x,y = delta_outer_layer.shape
		delta_outer_layer = delta_outer_layer.reshape(x,y)
		# print("delta outer layer",delta_outer_layer.shape[0])
		

		difference_hidden_layer = np.dot(delta_outer_layer,self.weights_[1].T)
		derivative_hidden_layer = layer_activations[0]*(1-layer_activations[0]) #sigmoid
		delta_hidden_layer =np.array([difference_hidden_layer*derivative_hidden_layer])
		a,x,y = delta_hidden_layer.shape
		delta_hidden_layer = delta_hidden_layer.reshape(x,y)
		# print("delta_hidden", delta_hidden_layer.shape[0])
		deltas.append(delta_hidden_layer)
		deltas.append(delta_outer_layer)
		
		return deltas
			
	def weight_update(self, deltas, layer_inputs, lr):
		'''Executes the gradient descent algorithm.
		"deltas" is return value of the backward pass step
		"layer_inputs" is a list containing the inputs for all layers (including the input layer)
		"lr" is the learning rate
		You just have to implement the simple weight update equation. 
		
		'''
		for i in range(2):
			self.weights_[i] -= lr * np.dot(deltas[i].T,layer_inputs[i]).T
			self.biases_[i] -= lr * sum(deltas[i])
	  
	   
		
	def predict(self, Xs):
		'''Returns the model predictions (output of the last layer) for the given "Xs".'''
		predictions = []
		
		for i in range(len(Xs)):
			preds = self.forward_pass(Xs[i,:].reshape((1, self.input_shape)))[-1]
			predictions.append(preds.reshape((self.output_shape)))
		return np.array(predictions)
	
	def evaluate(self, Xs, Ys):
		'''Returns appropriate metrics for the task, calculated on the dataset passed to this method.'''
		pred = self.predict(Xs)
		acc = self.accuracy(pred.argmax(axis=1), Ys.argmax(axis=1))
		loss = self.cross_entropy_loss(pred, Ys)
		return loss,acc

	def give_labels(self,listDirImages):
		'''Returns the images and labels from the listDirImages list after reading
		Hint: Use os.listdir(),os.getcwd() functions to get list of all directories
		in the provided folder. Similarly os.getcwd() returns you the current working
		directory. 
		For image reading use any library of your choice. Commonly used are opencv,pillow but
		you have to install them using pip
		"images" is list of numpy array of images 
		labels is a list of labels you read 
		'''
		images = []
		labels = []
		assignment_folder = os.getcwd()
		assignment_folder = str(assignment_folder)
	
		p = listDirImages+'.txt'
		for folders in os.listdir(assignment_folder):
			#print(folders)
			if folders == 'txt+files':
				current_folder = assignment_folder + "\\" + folders
				for files in os.listdir(current_folder):
					if files == 'txt files':
						new_directory = current_folder + '\\' + files
						for data in os.listdir(new_directory):
							if data == listDirImages + '-labels.txt':
								labels_folder = new_directory + "\\" + data
								file = open(labels_folder,'r')
								rows = file.read()
								for i in range(len(rows)):
									if rows[i] == '\n':
										continue
									labels.append(int(rows[i]))
								return labels

	def give_images(self,listDirImages):
		'''Returns the images and labels from the listDirImages list after reading
		Hint: Use os.listdir(),os.getcwd() functions to get list of all directories
		in the provided folder. Similarly os.getcwd() returns you the current working
		directory. 
		For image reading use any library of your choice. Commonly used are opencv,pillow but
		you have to install them using pip
		"images" is list of numpy array of images 
		labels is a list of labels you read 
		'''
		images = []
		labels = []
		assignment_folder = os.getcwd()
		assignment_folder = str(assignment_folder)
	
		p = listDirImages+'.txt'
		for folders in os.listdir(assignment_folder):
			#print(folders)
			if folders == 'txt+files':
				current_folder = assignment_folder + "\\" + folders
				for files in os.listdir(current_folder):
					if files == 'txt files':
						new_directory = current_folder + '\\' + files
						for data in os.listdir(new_directory):
							if data == listDirImages + '.txt':
								img_folder = new_directory + "\\" + data
								
								file=open(img_folder,'r')
								i=0
								while True:
									word=file.readline()
									if word=='':
										break
									images.append([])
									final_word = ""
									while True:
										if '[' in word:
											word=word[1:]
										if ']' in word:
											word=word[:-2]
											final_word=final_word+word.replace('\n', " ")
											break
										final_word=final_word+word.replace('\n', " ")
										word = file.readline()
									res = final_word.split()
									int_res = list(map(int,res))
									images[i] = int_res
									i = i + 1
								file.close()
								return images
	
	def generate_labels(self,labels):
		'''Returns your labels into one hot encoding array
		labels is a list of labels [0,1,2,3,4,1,3,3,4,1........]
		Ex-> If label is 1 then one hot encoding should be [0,1,0,0,0,0,0,0,0,0]
		Ex-> If label is 9 then one hot encoding shoudl be [0,0,0,0,0,0,0,0,0,1]
		Hint: Use sklearn one hot-encoder to convert your labels into one hot encoding array
		"onehotlabels" is a numpy array of labels. In the end just do np.array(onehotlabels).
		'''
		onehotencoder1 = OneHotEncoder(sparse=False)
		labels=labels.reshape(-1,1)
		labels = onehotencoder1.fit_transform(labels)
		
		return (np.array(labels))

	def save_weights(self,fileName):
		'''save the weights of your neural network into a file
		Hint: Search python functions for file saving as a .txt'''
		x = pickle.dumps(self.weights_)
		# y = pickle.dumps(self.biases_)
		file = open(fileName, 'wb')
		file.write(x)
		# file.write(y)
		file.close()
		
	def reassign_weights(self,fileName):
		'''assign the saved weights from the fileName to the network
		Hint: Search python functions for file reading
		'''
	def savePlot(self, file, t, a, l):
		'''function to plot the execution time versus learning rate plot
		You can edit the parameters pass to the savePlot function'''
		# plt.plot(t, a, 'b', label = f'learningRate: {l}') 
		plt.legend()
		plt.savefig(file)


	def read_labels(self, file):
		labels = []
		myfile = open(file, 'r')
		l = myfile.readlines()
		for rows in l:
			rows = rows.replace(',', '')
			rows = rows.replace('\n', '')
			labels.append(int(rows))
		labels = np.array(labels)
		return labels		


def main():

	nn = NeuralNetwork()
	start_time = time.time()
	print("Reading training data.....")
	images_train= nn.give_images("train")
	labels_train=nn.give_labels("train")

	t_array = []
	acc_array = []

	images_train = np.array(images_train)
	images_train = np.true_divide(images_train, 255)
	hot_encoded_labels_train = nn.generate_labels(np.array(labels_train))
	
	print("File Reading Completed")
	print('\n')
	print('\n')

	print('Training Model......')
	print('\n')
	hist = nn.fit(images_train, np.array(hot_encoded_labels_train), 2,t_array, acc_array, start_time, 0.2)

	# nn.savePlot('myplot.txt',t_array, acc_array, 0.2)

	print("Reading testing data.....")

	images_test= nn.give_images("test")
	labels_test=nn.give_labels("test")

	images_test = np.array(images_test)
	images_test = np.true_divide(images_test, 255)

	
	hot_encoded_labels_test = nn.generate_labels(np.array(labels_test))
	print('Reading for testing data Completed..... ')

	
	nn.save_weights('netWeights.txt')
	loss, accuracy = nn.evaluate(images_test, np.array(hot_encoded_labels_test))
	acc = (accuracy/float(10000))*100
	print('Accuracy for testing data',acc,'%')
	



	
	#print(len(weights_))
	

	#xyz.__init_weights_(3)

		
main()


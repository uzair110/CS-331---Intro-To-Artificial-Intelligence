import numpy as np 
import pandas as pd
import sys



class bernoulli_naive_bayes(object):

	def __init__(self):
		self.abnormal_prob = 0.0
		self.abnormal = 0
		self.normal = 0
		self.normal_prob = 0.0
		self.zeros_in_normal = 0.0
		self.ones_in_normal = 0.0
		self.zeros_in_abnormal = 0.0
		self.ones_in_abnormal = 0.0
		self.binary = []
		


	def read_file(self, file):
		data_ones = []
		data_zeros= []
		data_final_ones = []
		data_final_zeros= []
	
		myfile = open(file, 'r')
		l = myfile.readlines()
		for rows in l:
			rows = rows.replace(',', '')
			rows = rows.replace('\n', '')
			self.binary.append(int(rows[0]))
			if int(rows[0]) == 1:
				sp_1 = []
				for i in range(1, len(rows)):
					sp_1.append(int(rows[i]))
				data_ones.append(sp_1)
			if int(rows[0]) == 0:
				sp_0 = []
				for i in range(1,len(rows)):
					sp_0.append(int(rows[i]))
				data_zeros.append(sp_0)


		myfile.close()
		data_ones = np.array(data_ones)
		data_zeros = np.array(data_zeros)
		return data_ones, data_zeros, np.array(self.binary)

	def read_test_file(self, file):
		data = []
		classes = []
	
		myfile = open(file, 'r')
		l = myfile.readlines()
		for points in l:
			sp = [] #singlepoint
			points = points.replace(',', '')
			points = points.replace('\n', '')
			classes.append(int(points[0])) #classifier (1 or 0)
			for i in range(1,len(points)):
				singlepoint = int(points[i])
				sp.append(int(singlepoint))
			data.append(sp)
		data = np.array(data)
		myfile.close()
		return data, np.array(classes)

	def priors(self, binary):
		normal_prob = np.sum(self.binary)/len(self.binary)
		abnormal_prob = 1 - normal_prob
		return normal_prob, abnormal_prob

	def likelihood_normal(self, tests_1, binary):
		one_normal = np.sum(tests_1)/len(tests_1)
		zero_normal = 1 - one_normal
		return (zero_normal, one_normal)

	def likelihood_abnormal(self, tests_0, binary):
		one_abnormal = np.sum(tests_0)/len(tests_0)
		zero_abnormal = 1 - one_abnormal
		return (zero_abnormal, one_abnormal)

	def check_accuracy(self, tests, probabilities, normal_prob, abnormal_prob):
		
		abnormal_test = 1
		for i in range(len(tests)):
			if tests[i] == 1:
				abnormal_test *= probabilities[i][1]
			if tests[i] == 0:
				abnormal_test *= probabilities[i][0]
		abnormal_test *= abnormal_prob

		normal_test = 1
		for i in range(len(tests)):
			if tests[i] == 1:
				normal_test *= probabilities[i][3]
			if tests[i] == 0:
				normal_test *= probabilities[i][2]
		normal_test *= normal_prob
		
		if normal_test > abnormal_test:
			return 1
		else:
			return 0


def main():

	    xyz = bernoulli_naive_bayes()
	    data_1, data_0, classifier = xyz.read_file(sys.argv[1] + '.txt')
	  
	    print('##############')
	    no_of_test = data_1.shape[1]
	    print("Starting to Train on {} data points...".format(len(data_1)+len(data_0)))
	    n, a = xyz.priors(classifier)
	    
	    data1 = data_1.transpose()
	    data0 = data_0.transpose()
	   
	    probabilities=[]
	    p5 = []
	  	
	    for i in range(no_of_test):
	    	zeros_normal, one_normal = xyz.likelihood_normal(data1[i], classifier)
	    	zeros_abnormal, one_abnormal = xyz.likelihood_abnormal(data0[i], classifier)
	    	p5 = zeros_abnormal, one_abnormal,zeros_normal, one_normal
	    	probabilities.append(np.array(p5))

	    probabilities = np.array(probabilities)
	    print("Training Complete")
	  
	    matches = 0
	    print('\n')
	    print('\n')

	    
	    test_data, classifier_test = xyz.read_test_file(sys.argv[2] + '.txt')
	    print("Testing on {} data points...".format(len(test_data)))
	   
	    no_of_test_data = test_data.shape[0]
	    for i in range(no_of_test_data):
	    	acc = xyz.check_accuracy(test_data[i], probabilities,n, a)
	    	if acc == classifier_test[i]:
	    		matches +=1
	    print('Testing Complete')
	    
	    accuracy = matches/no_of_test_data
	    print('Total accuracy =',accuracy*100)
	    print('##############')
	   
	    
main()
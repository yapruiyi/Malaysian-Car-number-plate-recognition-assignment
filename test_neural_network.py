import numpy as np
from PIL import Image
import math
import random
import cv2 as cv
from optimized_training import NeuralNetwork

s1 = NeuralNetwork(60, 20, 0.5) # first argument is number of hidden neurons, second is number of output neurons
s1.weights_ji = np.loadtxt("weights_ji_aftertrain.txt")
s1.weights_kj = np.loadtxt("weights_kj_aftertrain.txt")
s1.bias_j = np.loadtxt("bias_j_aftertrain.txt")
s1.bias_k = np.loadtxt("bias_k_aftertrain.txt")
flag = False
imgchars = "0" + "1" + "2" + "3" + "4" + "5" + "6" + "7" + "8" + "9" + "B" + "F" + "L" + "M" + "P" + "Q" + "T" + "U" +  "V" + "W"

    
myfiles = s1.Read_Files()

print("Testing Phase")
    
output = []
testing_data = myfiles[2]
testing_data_targets = myfiles[3]

for i in range(len(testing_data)):
    arr = testing_data[i]
    s1.input = arr
    s1.Forward_Input_Hidden()
    output.append([k for k in s1.Forward_Hidden_Output()])

s1.accuracy(output, testing_data_targets, len(testing_data))
s1.save_weights_bias()




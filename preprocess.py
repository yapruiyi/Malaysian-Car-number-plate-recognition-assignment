"""
This file generates the following files to be used by the optimized_training.py file
testing_data.txt
testing_data_targets.txt
training_data.txt
training_data_targets.txt
"""

import numpy as np
import cv2 as cv


temp = []
temp2 = []
training_data = []
training_data_targets = []
testing_data = []
testing_data_targets = []
threshold = 1
imgvalues = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "B", "F", "L", "M", "P", "Q", "T", "U", "V", "W"] # uncomment this to test only all characters, make sure to change number of output neurons to 20
#imgvalues = ["B", "F", "L", "M", "P", "Q", "T", "U", "V", "W"] # uncomment this to test only alphabets, make sure to change number of output neurons to 10
#imgvalues = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] # uncomment this to test only numbers, make sure to change number of output neurons to 10

for x in range(8):
    for y in range(len(imgvalues)):
        img = cv.imread("Training Image/" + imgvalues[y] + "0" + str(x) + ".png")
        img = cv.resize(img,(20,40))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.Canny(img, 150, 200)
        # cv.imshow("myimg", img)
        # cv.waitKey()
        img_arr = np.array(img)
        img_arr = np.where(img_arr > threshold, 1, 0)
        arr = img_arr.flatten()
        temp.append(arr.tolist())
        temp2.append(y)
    training_data += [i for i in temp]
    training_data_targets += [i for i in temp2]
    temp = []
    temp2 = []


temp = []
temp2 = []
training_data = np.array(training_data)
training_data_targets = np.array(training_data_targets)
print(len(training_data))
np.savetxt("training_data.txt", training_data)
np.savetxt("training_data_targets.txt", training_data_targets)


for x in range(8, 10, 1):
    for y in range(len(imgvalues)):
        img = cv.imread("Testing Image/" + imgvalues[y] + "0" + str(x) + ".png")
        img = cv.resize(img,(20, 40))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.Canny(img, 100, 150)
        img_arr = np.array(img)
        img_arr = np.where(img_arr > threshold, 1, 0)
        arr = img_arr.flatten()
        temp.append(arr.tolist())
        temp2.append(y)
    testing_data += [i for i in temp]
    testing_data_targets += [i for i in temp2]
    temp = []
    temp2 = []


testing_data = np.array(testing_data)
testing_data_targets = np.array(testing_data_targets)
print(len(testing_data))
np.savetxt("testing_data.txt", testing_data)
np.savetxt("testing_data_targets.txt", testing_data_targets)



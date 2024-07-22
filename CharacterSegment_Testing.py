"""
This file should be run after optimized_training.py or after test_neural_network.py

code referenced from http://dangminhthang.com/knowledge-sharing/characters-segmentation-and-recognition-for-vehicle-license-plate/
we have modified and added, our own conditions for our use case
"""
import os
import functools
import cv2 as cv
import numpy as np
from optimized_training import NeuralNetwork

img_rectangle = []
carplate_chars = ["VBU3878" , "VBT2597" , "WTF6868" , "PLW7969" , "BPU9859" , "BMT8628" , "BMB8262" , "PPV7422" , "BQP8189" , "WUM207"]

for i in range(1,11):
    img = cv.imread("Carplate Image/"+ str(i) +  ".jpg")
 
    img = cv.resize(img, (357,107))
    
    # cv.imshow("img",img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", gray)
    # Apply Gaussian blurring and thresholding 
    # to reveal the characters on the license plate
    blurred = cv.GaussianBlur(gray, (9, 9), 0)
    _, threshold = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)
    
    # Perform connected components analysis on the thresholded image and
    # initialize the mask to hold only the components we are interested in
    _, labels = cv.connectedComponents(threshold)
    mask = np.zeros(threshold.shape, dtype="uint8")

    # Set lower bound and upper bound criteria for characters
    total_pixels = img.shape[0] * img.shape[1]
    lower = total_pixels // 70 # heuristic param, can be fine tuned if necessary
    upper = total_pixels // 10 # heuristic param, can be fine tuned if necessary

    # Loop over the unique components
    for (i, label) in enumerate(np.unique(labels)):
        # If this is the background label, ignore it
        if label == 0:
            continue
    
        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(threshold.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv.countNonZero(labelMask)
    
        # If the number of pixels in the component is between lower bound and upper bound, 
        # add it to our mask
        if numPixels > lower and numPixels < upper:
            mask = cv.add(mask, labelMask)

    # Find contours and get bounding box for each contour
    cnts, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv.boundingRect(c) for c in cnts]


    # Sort the bounding boxes from left to right, top to bottom
    # sort by X first, and then sort by Y if Xs are similar
    def compare(rect1, rect2):
        if abs(rect1[0] - rect2[0]) > 10:
            return rect1[0] - rect2[0]
        else:
            return rect1[1] - rect2[1]

    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )

    average_width = []
    
    # Draw bounding boxes on the image
    for (x, y, w, h) in boundingBoxes:
        #print(x, x+w, w)
        if average_width == []:
            average_width.append(w)
            #cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 0) #357 107
            if (y > 0 and y + h < 357) and (x > 0 and x + w + 1 < 357):
                cropped_img = img[y - 1:y + h + 1, x - 1:x + w + 1 + 1]
            elif y == 0:
                cropped_img = img[y:y + h + 1, x - 1:x + w + 1 + 1]
            else:
                cropped_img = img[y:y + h, x:x + w + 1]
            img_rectangle.append(cropped_img)
        elif w > sum(average_width)//len(average_width) + 5:
            print("higher than mean")
            average_width.append(w // 2 )
            #cv.rectangle(img, (x, y), (x + (w//2), y + h), (0, 0, 0), 0)
            #cv.rectangle(img, (x, y), ((x + (w//2)) + (w//2 + 1), y + h), (0, 0, 0), 2)
            if (y > 0 and y + h < 357) and (x > 0 and x + (w // 2) < 357):
                cropped_img = img[y - 1:y + h + 1, x - 1:x + (w // 2) + 1]
            else:
                cropped_img = img[y:y + h, x:x + (w // 2)]

            if (y > 0 and y + h < 357) and (x + (w // 2) > 0 and ((x + (w//2)) + (w//2 + 1) + 1)< 357):
                cropped_img2 = img[y - 1:y + h + 1, x + (w // 2) - 1: ((x + (w//2)) + (w//2 + 1) + 1 + 1)]
            else:
                cropped_img2 = img[y:y + h, x + (w // 2): ((x + (w//2)) + (w//2 + 1) + 1)]
            img_rectangle.append(cropped_img)
            img_rectangle.append(cropped_img2)
        elif w < sum(average_width)//len(average_width) - 5:
            print("lower than mean")
            average_width.append(w * 1.5 )
            #cv.rectangle(img, (x, y), (int(x + (w*1.5)), y + h), (0, 0, 0), 0)
            if (y > 0 and y + h < 357) and (x> 0 and x + 1 + int(w * 1.5) < 357):
                cropped_img = img[y - 1:y + h + 1, x - 1:x + 1 + int(w * 1.5) + 1]
            else:
                cropped_img = img[y:y + h, x:x + 1 + int(w * 1.5)]
            img_rectangle.append(cropped_img)
            # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            average_width.append(w)
            #cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 0)
            if (y > 0 and y + h < 357) and (x > 0 and x + w + 1 < 357):
                cropped_img = img[y - 1:y + h + 1, x - 1:x + w + 1 + 1]
            else:
                cropped_img = img[y:y + h, x:x + w + 1]
            
            img_rectangle.append(cropped_img)

path = "segmentedcarplatechars"   
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)
    count = 0   
    for char in img_rectangle:
        cv.imwrite( path + "/" + str(count) + ".jpg", char)
        count += 1

    
    # Display the image with bounding boxes
    # cv.imshow("Image with Bounding Boxes", img)


    # cv.waitKey()
#     cv.destroyAllWindows()

# cv.imshow("8",img_rectangle[5])
# cv.waitKey()


# below here is testing with the neural network
imgvalues = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "B", "F", "L", "M", "P", "Q", "T", "U", "V", "W"]
temp = []
temp2 = []
testing_data = []
testing_data_targets = [] 
newthreshold = 1

carplate_chars_concatenated = "".join(carplate_chars)

for i in range(len(carplate_chars_concatenated)):
    testing_data_targets += [imgvalues.index(carplate_chars_concatenated[i])]

print(testing_data_targets)

for i in range(len(img_rectangle)):
    newimg = cv.resize(img_rectangle[i], (20,40))
    newimg = cv.cvtColor(newimg, cv.COLOR_BGR2GRAY)
    newimg = cv.Canny(newimg, 50, 110)
    # cv.imshow("myimg", newimg)
    # cv.waitKey()


    
    img_arr = np.array(newimg)
    img_arr = np.where(img_arr > newthreshold, 1, 0)
    myarr = img_arr.flatten()
    testing_data.append(myarr)

print(len(testing_data))


print("Testing Phase")
s1 = NeuralNetwork(50, 20, 0.5) # first argument is number of hidden neurons, second is number of output neurons
s1.weights_ji = np.loadtxt("weights_ji_aftertrain.txt")
s1.weights_kj = np.loadtxt("weights_kj_aftertrain.txt")
s1.bias_j = np.loadtxt("bias_j_aftertrain.txt")
s1.bias_k = np.loadtxt("bias_k_aftertrain.txt")
output = []

for i in range(len(testing_data)):
    arr = testing_data[i]
    s1.input = arr
    s1.Forward_Input_Hidden()
    output.append([k for k in s1.Forward_Hidden_Output()])

s1.accuracy(output, testing_data_targets, len(testing_data), carplatechars= carplate_chars_concatenated)
#s1.saveweightbias()

import numpy as np


class NeuralNetwork:
    def __init__(self, num_hidden_neuron, num_out_neuron, initial_learning_rate):
        self.input = None
        self.weights_ji = None
        self.weights_kj = None
        self.bias_j = None
        self.bias_k = None
        self.target = None
        self.out_j = None
        self.out_k = None
        self.error = None
        self.delta_k = None
        self.delta_j = None
        self.learning_rate = initial_learning_rate
        self.epoch = None

        self.num_hidden_neuron = num_hidden_neuron
        self.num_out_neuron = num_out_neuron

    def Weight_Initialization(self):
        np.random.seed(5000)
        self.weights_ji = np.random.uniform(-0.5, 0.5, size=(self.num_hidden_neuron, self.input.shape[0]))
        self.weights_kj = np.random.uniform(-0.5, 0.5, size=(self.num_out_neuron, self.num_hidden_neuron))
        self.bias_j = np.random.uniform(0, 1, size=(self.num_hidden_neuron,))
        self.bias_k = np.random.uniform(0, 1, size=(self.num_out_neuron,))

    def Forward_Input_Hidden(self):
        self.out_j = 1 / (1 + np.exp(-(np.dot(self.weights_ji, self.input) + self.bias_j)))

    def Forward_Hidden_Output(self, target = None, epoch_count = None, batch = None):
        self.out_k = 1 / (1 + np.exp(-(np.dot(self.weights_kj, self.out_j) + self.bias_k)))

        #uncomment below to check target output of each individual characters during training
        
        if epoch_count != None and epoch_count == self.epoch - 1:
            alphabets = ["B", "F", "L", "M", "P", "Q", "T", "U", "V", "W"]
            
            if self.out_k[int(target)] >= 0.9:
                copy = self.out_k.tolist()
                copy.pop(int(target))
                for i in range(len(copy)):
                    if copy[i] > 0.1:
                        print("met the target of more than 0.9 but other values are more than 0.1")
                        if target > 9:
                            print("batch = " + str(batch) + " failed target output of " + alphabets[int(target) - 10] + " = " + str(self.out_k))
                        else:
                            print("batch = " + str(batch) + " failed target output of " + str(int(target)) + " = " + str(self.out_k))
                        break
            else:
                print("did not met the target of more than 0.9")
                if target > 9:
                    print("batch = " + str(batch) + " failed target output of " + alphabets[int(target) - 10] + " = " + str(self.out_k))
                else:
                    print("batch = " + str(batch) + " failed target output of " + str(int(target)) + " = " + str(self.out_k))

        return self.out_k

    def Check_for_End(self, count):
        self.epoch = 500
        if count == self.epoch:
            return False
        else:
            return True

    def Error_Correction(self):
        self.error = 0.5 * (self.target - self.out_k) ** 2

    def Weight_Bias_Correction_Output(self):
        self.delta_k = (self.out_k - self.target) * self.out_k * (1 - self.out_k)

    def Weight_Bias_Correction_Hidden(self):
        self.delta_j = np.dot(self.weights_kj.T, self.delta_k) * self.out_j * (1 - self.out_j)

    def Weight_Bias_Update(self):
        self.weights_kj -= self.learning_rate * np.outer(self.delta_k, self.out_j)
        self.weights_ji -= self.learning_rate * np.outer(self.delta_j, self.input)
        self.bias_k -= self.learning_rate * self.delta_k
        self.bias_j -= self.learning_rate * self.delta_j

    def train(self, training_data, training_data_targets):
        self.input = training_data[0]
        self.target = np.zeros(self.num_out_neuron)
        epoch_count = 0
        cycle = 1
        batch = 0

        self.Weight_Initialization()

        while self.Check_for_End(epoch_count):
            print("Epoch:", epoch_count + 1)  # Print the current epoch
            for data, target in zip(training_data, training_data_targets):
                self.input = data
                self.target[:] = 0
                self.target[int(target)] = 1

                if (cycle == 21):
                    batch += 1
                    cycle = 1

                self.Forward_Input_Hidden()
                self.Forward_Hidden_Output(target, epoch_count, batch)
                self.Weight_Bias_Correction_Output()
                self.Weight_Bias_Correction_Hidden()
                self.Weight_Bias_Update()
                cycle += 1
            batch = 0
            cycle = 1
            epoch_count += 1

            self.update_learning_rate(epoch_count)
            
        self.save_weights_bias()

    def update_learning_rate(self, epoch):
        self.learning_rate = 0.5 / (1 + 0.01 * epoch)  # You can adjust the decay rate

    def save_weights_bias(self):
        np.savetxt("weights_ji_aftertrain.txt", self.weights_ji)
        np.savetxt("weights_kj_aftertrain.txt", self.weights_kj)
        np.savetxt("bias_j_aftertrain.txt", self.bias_j)
        np.savetxt("bias_k_aftertrain.txt", self.bias_k)

    def accuracy(self, output, targets, numberoftestingimg, carplatechars = None):
        number_of_correct_testing_img = 0
        number_of_correct_char_lst = [0] * 20
        frequencyofeachcarplatechars = [1, 1, 7, 1, 1, 2, 5, 5, 10, 6, 7, 1, 1, 3, 5, 1, 3, 3, 3, 3]
        alphabets = ["B", "F", "L", "M", "P", "Q", "T", "U", "V", "W"]
        characters = "0123456789BFLMPQTUVW"
        res = ""


        for arr,val in zip(output, targets):
            highest_output_index = np.argmax(arr)
            if highest_output_index == int(val):
                number_of_correct_char_lst[int(val)] += 1
                number_of_correct_testing_img += 1
            if highest_output_index > 9:
                res += str(alphabets[highest_output_index - 10])
            else:
                res += str(highest_output_index)
                # print("character = " + str(arr))
                # print("val = " + str(val))

        if carplatechars == None:
            for index, char in enumerate(characters):
                print("numbers of " + char + " tested to be correct = " + str(number_of_correct_char_lst[index]) + ", accuracy of character " + char + " is = " + str((number_of_correct_char_lst[index]/ 2) * 100) + "%")
            
        if carplatechars != None:
            for index, char in enumerate(characters):
                print("numbers of " + char + " tested to be correct = " + str(number_of_correct_char_lst[index]) + ", accuracy of character " + char + " is = " + str((number_of_correct_char_lst[index]/ frequencyofeachcarplatechars[index]) * 100) + "%")

            n = 7
            print("carplate images are classified as = " + str([res[i:i+n] for i in range(0, len(res), n)]))

        accuracy = number_of_correct_testing_img / numberoftestingimg

        #print("output = " + str(output))
        print("Total number of tested image = " + str(numberoftestingimg))
        print("Number of correct tested image = " + str(number_of_correct_testing_img))
        print("Total accuracy = " + str(accuracy * 100) + "%")
        return accuracy

    @staticmethod
    def Read_Files():
        return (
            np.loadtxt("training_data.txt"),
            np.loadtxt("training_data_targets.txt"),
            np.loadtxt("testing_data.txt"),
            np.loadtxt("testing_data_targets.txt")
        )

if __name__ == '__main__':
    s1 = NeuralNetwork(50, 20, 0.5)  # First argument is the number of hidden neurons, second is the number of output neurons

    myfiles = s1.Read_Files()
    training_data = myfiles[0]
    training_data_targets = myfiles[1]

    s1.train(training_data, training_data_targets)
    

# Author: Valter Nunez
# Credits: Jason Brownlee from Machine Learning Mastery
# Dataset: Swedish Auto Insurance Dataset: https://www.math.muni.cz/~kolacek/docs/frvs/M7222/data/AutoInsurSweden.txt

#Imports
from math import sqrt
from csv import reader
from random import randrange
from random import seed

#Import Dataset from CSV
def load_csv(filename):
    dataset = list() #Initialize
    with open(filename, 'r', encoding='utf-8-sig') as file:#Needed the encoding for some strange UTF reason.
        csv = reader(file)
        for row in csv: #Add row 1 by 1
            if not row:   #If the row is empty, skip it.
                continue
            dataset.append(row)
    return dataset

#Cast values from string to float from the dataset .csv
def toFloat(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip()) #cast

#Get Train and Test dataset
def train_test_split(dataset, split):
    train = list() #Initialization.
    train_size = split * len(dataset) #You multiply for the split percentage to avoid doing it later. We already know the size, we are just missing knowing which of those will be where. Something like that haha
    dataset_copy = list(dataset) #Copy the dataset
    while len(train) < train_size:
        index = randrange(len(dataset_copy)) #Random asigning of rows to test and train
        train.append(dataset_copy.pop(index)) #Take from copied dataset to train dataset
    return train, dataset_copy #The copied dataset becomes the test dataset

#Calculate Mean Square error. Used this one because it penalizes more than Mean Absolute Error (larger error at least)
def mse_metric(actual, predicted):
    sum_error = 0.0 #Initialization
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual [i] #part of the formula. This value will be squared later and added by the sumatory.
        sum_error += (prediction_error ** 2) # **Elevates to 2nd power the error
    mean_error = sum_error / float(len(actual)) #Actual result divides the sumatory of all before and divides it by the number of total predictions.
    return sqrt(mean_error)

#Will evauluate the algorithm.  We will obtain the Mean Squared Error out of the calculatons of this.
def evaluate_algorithm(dataset, algorithm, split, *args):
    train, test = train_test_split(dataset, split) #Division of test and training
    test_set = list() #initialization
    for row in test: #copy the test set
        row_copy = list(row) #Make the actual copy
        row_copy[-1] = None #On this dataset, the last column is result, so we skip it.
        test_set.append(row_copy) #Append to the test set in order to do the test set
    predicted = algorithm(train, test_set, *args) #Store the prediceted result of the algorith, in the model.
    actual = [row[-1 ] for row in test] #Store the actual result of the algorithm in the model
    print('Predicted data: ')
    print(predicted)
    print('Actual data: ')
    print(actual)
    mse = mse_metric(actual, predicted) #Check the results of comparing both the training and the testing set.
    return mse

#Basic mean function that needs no explanation, honestly.
def mean(values):
    return sum(values) / float(len(values)) #Basic Mean function

# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values]) #Basic variance formula

#Calculate covariance in order to get relation between data
def covariance(x, mean_x, y, mean_y):
    res = 0.0 #Initalize the data
    for i in range(len(x)): #Sumatory
        res += (x[i] - mean_x) * (y[i] - mean_y) #Plain covariance formula, honestly.
    return res

#Calculations of coefficients. These are B1 and B0 from the basic Linear Regression formula.
def coefficients(dataset):
    x = [row[0] for row in dataset] #Intialization with the dataset data on x axis
    y = [row[1] for row in dataset] #Intialization with the dataset data on y axis
    x_mean, y_mean = mean(x), mean(y) #Calculations of x and y means.
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean) #You need to get covariance(x,y)/variance(x). You send the mean too so you dont calculate it more than it is necessary.
    b0 = y_mean - b1 * x_mean #Basic formula for B0 need B1 already calculated.
    return [b0, b1]

#Now that we have everything we need, we got to create the linea regression.
def simple_linear_regression(train, test):
	predictions = list() #Initialization.
	b0, b1 = coefficients(train) #Obtain B0 and B1 from the dataset given.
	for row in test: #Go through each entry
		res = b0 + b1 * row[0] #Literal linea regression formula.     y=b0+b1*x
		predictions.append(res) #Append the results of the predictions on the test part of the dataset.
	return predictions #See if we are doing things OK, pretty much.


seed(1) #Initialize randoms in the program.
filename = 'insurance.csv' #Import CSV to filename
dataset = load_csv(filename) #Load dataset from the CSV
for i in range(len(dataset[0])):
	toFloat(dataset, i) #Transform from String to Float all the numbers in the dataset.
split = 0.7 #70 training / 30 Testing
mse = evaluate_algorithm(dataset, simple_linear_regression, split) #Obtain the Mean Squared Error from the evualuation of the algorithm with the test and training dataset.
print('Mean Squared Error:  %.3f ' % mse) #Print the Mean Squared Error


'''
*******************************
Author: Raza, Manaswini, Rami, Lithika
u123456 Assessment 1_Program1_(a) 22/ 11/2024
Programming:
*******************************
'''        



import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
import pickle
from Analysis import CSVReader
import numpy as np

class MachineLearning:

    def __init__(self):
        pass   
        #Linear Regression
    def LinearRegression(self, filePath, ): # Pasha's Code

        csvReader = CSVReader()
        Regression = LinearRegression()
        dataFrame = csvReader.readCsv(filePath)
        X = dataFrame['Price']
        Y = dataFrame['ppi']
        Regression.fit(X.values.reshape(-1, 1), Y) # pass single collumn to data frame
        print(Regression.score(X.values.reshape(-1, 1), Y)) # Get the coefficient to determine the gradient

        # below to be moved later to gui, just for testing purposes
        plt.scatter(X,Y)
        plt.plot(X, Regression.predict(X.values.reshape(-1,1)), color = 'red')
        plt.xlabel('Price')
        plt.ylabel('PPI')
        plt.title('Linear Regression')
        plt.show()

    def LinearRegressionNoGUI(self, filePath, input): # Pasha's Code

        csvReader = CSVReader()
        Regression = LinearRegression()
        dataFrame = csvReader.readCsv(filePath)
        X = dataFrame['Price']
        Y = dataFrame['ppi']
        Regression.fit(X.values.reshape(-1, 1), Y) # pass single collumn to data frame
        print(Regression.score(X.values.reshape(-1, 1), Y)) # Get the coefficient to determine the gradient
        output = Regression.predict([[input]])
        return output


    def DecisionTree(self, filePath): #Rami's and Pasha's code

        csvReader = CSVReader()
        dataFrame = csvReader.readCsv(filePath)
        X = dataFrame['Price']
        Y = dataFrame['ppi']
        
        regModel = DecisionTreeRegressor(max_depth=5,criterion='friedman_mse') #implementing good range
        print(regModel) # Printing all the parameters of Decision Tree
        DT=regModel.fit(X.values.reshape(-1, 1), Y)  # Creating the model on our training Data
        prediction=DT.predict(X.values.reshape(-1,1))
        print('R^2 Value:',metrics.r2_score(Y, DT.predict(X.values.reshape(-1,1)))) # Measuring how well training data fits
       

        # Plot the data - Pasha
        plt.scatter(X, Y, label='Actual Data')
        plt.plot(X, prediction, label='Predicted Data', color='red')
        plt.xlabel('Price')
        plt.ylabel('ppi')
        plt.title('Decision Tree Regression')
        plt.legend()
        plt.show()
    
    #random forest
    def RandomForest(self, filePath):
        csvReader = CSVReader()
        dataFrame = csvReader.readCsv(filePath)
        X = dataFrame[['Price']]  # Independent variable
        Y = dataFrame['ppi']  # Dependent variable
        
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X, Y)
        
        predictions = model.predict(X)
        print("Random Forest R2 score:", r2_score(Y, predictions))
        print("Random Forest RMSE:", mean_squared_error(Y, predictions, squared=False))
        
        # Optional: Plot results for testing
        plt.scatter(X, Y)
        plt.plot(X, predictions, color='green')
        plt.xlabel('Price')
        plt.ylabel('PPI')
        plt.title('Random Forest Regression')
        plt.show()

    #AdaBoost
    def AdaBoost(self, filePath):
        csvReader = CSVReader()
        dataFrame = csvReader.readCsv(filePath)
        X = dataFrame[['Price']]  # Independent variable
        Y = dataFrame['ppi']  # Dependent variable
        
        model = AdaBoostRegressor(n_estimators=50, random_state=0)
        model.fit(X, Y)
        
        predictions = model.predict(X)
        print("AdaBoost R2 score:", r2_score(Y, predictions))
        print("AdaBoost RMSE:", mean_squared_error(Y, predictions, squared=False))
        
        # Optional: Plot results for testing
        plt.scatter(X, Y)
        plt.plot(X, predictions, color='blue')
        plt.xlabel('Price')
        plt.ylabel('PPI')
        plt.title('AdaBoost Regression')
        plt.show()
    
    
    def SVMRegressor(self, filePath):
        csvReader = CSVReader()
        dataFrame = csvReader.readCsv(filePath)
        X = dataFrame[['Price']]  # Independent variable
        Y = dataFrame['ppi']  # Dependent variable

        model = SVR(kernel='rbf')  # 'linear', 'poly', 'rbf', 'sigmoid
        model.fit(X, Y)

        predictions = model.predict(X)
        print("SVM R2 score:", r2_score(Y, predictions))
        print("SVM RMSE:", np.sqrt(mean_squared_error(Y, predictions)))

        # I implemented a linear line for the SVM Regression Graph instead for you -Pasha
        plt.scatter(X, Y)
        z = np.polyfit(X.values.flatten(), Y.values, 1)
        p = np.poly1d(z)
        plt.plot(X, p(X.values.flatten()), "r--")
        plt.xlabel('Price')
        plt.ylabel('PPI')
        plt.title('SVM Regression')
        plt.show()


        
    

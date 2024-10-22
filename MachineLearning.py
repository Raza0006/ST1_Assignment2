
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
import tkinter as tk
from tkinter import messagebox
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
'''

#Best model
    def selectBestModel(self, filePath):
        csvReader = CSVReader()
        dataFrame = csvReader.readCsv(filePath)
        X = dataFrame[['Price']]  # Independent variable
        Y = dataFrame['ppi']  # Dependent variable
        
        models = {
            'Linear Regression': self.LinearRegression(filePath),
            'Decision Tree': self.DecisionTree(filePath),
            'Random Forest': self.RandomForest(filePath),
            'AdaBoost': self.AdaBoost(filePath),
            'SVM': self.SVMRegressor(filePath)
        }
        
        bestModel = None
        bestScore = float('-inf')  # Initialize with a very low value for R-squared comparison
        bestRmse = float('inf')    # Initialize with a very high value for RMSE comparison
        
        for modelName, modelPredictions in models.items():
            r2 = r2_score(Y, modelPredictions)
            rmse = mean_squared_error(Y, modelPredictions, squared=False)
            
            print(f'{modelName} R2 score: {r2}')
            print(f'{modelName} RMSE: {rmse}')
            
            # Choose the model with the highest R2 score
            if r2 > bestScore:
                bestModel = modelName
                bestScore = r2
            
            # Optionally, you could use RMSE to pick the model with the lowest error instead
            # if rmse < best_rmse:
            #    best_model = model_name
            #    best_rmse = rmse
        
        print(f'Best model is: {bestModel} with R2 score: {bestScore}')
        return bestModel


# Load the serialized model (Make sure best_model.pkl exists in the same directory)
    def loadModel():
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model

# Function to predict PPI based on the input price
    def predictPPI():
        try:
            price = float(price_entry.get())  # Not going to work as of yet. -Pasha
            model = MachineLearning.loadModel() # Fixed -Pasha
            prediction = model.predict([[price]])  # Predict the PPI based on the price
            messagebox.showinfo("Prediction", f"Predicted PPI: {prediction[0]}")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number for the price.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

        pass # Delete once implemented with nicegui
        
        #Commented out, we aren't using Tkinter. I'll be using NiceGUI. -Pasha

        # Function to trigger the prediction
        # Create the Tkinter window
        root = tk.Tk()
        root.title("Mobile Price Prediction")
        
        # Input label and entry for price
        tk.Label(root, text="Enter Price:").grid(row=0, column=0, padx=10, pady=10)
        price_entry = tk.Entry(root)
        price_entry.grid(row=0, column=1, padx=10, pady=10)
        
        # Button to trigger the prediction
        predict_button = tk.Button(root, text="Predict PPI", command=predict_ppi)
        predict_button.grid(row=1, column=1, padx=10, pady=10)
        
        # Run the Tkinter event loop
        root.mainloop()
        '''

        
    

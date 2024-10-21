import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from Analysis import CSVReader

class MachineLearning:

    def __init__(self):
        pass   
        #Linear Regression
    def LinearRegression(self, filePath): # Pasha's Code
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
        #Decision Tree
    def DecisionTree(self, filePath) #Rami 
        regModel = DecisionTreeRegressor(max_depth=5,criterion='friedman_mse') #implementing good range
        print(regModel) # Printing all the parameters of Decision Tree
        DT=RegModel.fit(X_train,y_train)  # Creating the model on our training Data
        prediction=DT.predict(X_test)
        print('R2 Value:',metrics.r2_score(y_train, DT.predict(X_train)) # Measuring how well training data fits

        %matplotlib inline #plotting most important columns
        feature_importances = pd.Series(DT.feature_importances_, index=Predictors)
        feature_importances.nlargest(10).plot(kind='barh')
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
    #SVMRegressor
    def SVMRegressor(self, filePath):
        csvReader = CSVReader()
        dataFrame = csvReader.readCsv(filePath)
        X = dataFrame[['Price']]  # Independent variable
        Y = dataFrame['ppi']  # Dependent variable
        
        model = SVR(kernel='rbf')  # You can experiment with different kernels like 'linear', 'poly', etc.
        model.fit(X, Y)
        
        predictions = model.predict(X)
        print("SVM R2 score:", r2_score(Y, predictions))
        print("SVM RMSE:", mean_squared_error(Y, predictions, squared=False))
        
        # Optional: Plot results for testing
        plt.scatter(X, Y)
        plt.plot(X, predictions, color='purple')
        plt.xlabel('Price')
        plt.ylabel('PPI')
        plt.title('SVM Regression')
            
            
        
        
    

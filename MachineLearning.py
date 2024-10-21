import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from Analysis import CSVReader

class MachineLearning:

    def __init__(self):
        pass    
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
       
        '''
        %matplotlib inline #plotting most important columns
        feature_importances = pd.Series(DT.feature_importances_, index=X.columns)
        feature_importances.nlargest(10).plot(kind='barh')
        '''


#Saved just in case
'''
    def DecisionTree(self, filePath): #Rami 

        csvReader = CSVReader()
        dataFrame = csvReader.readCsv(filePath)
        X = dataFrame['Price']
        Y = dataFrame['ppi']
        
        regModel = DecisionTreeRegressor(max_depth=5,criterion='friedman_mse') #implementing good range
        print(regModel) # Printing all the parameters of Decision Tree
        DT=RegModel.fit(X_train,y_train)  # Creating the model on our training Data
        prediction=DT.predict(X_test)
        print('R2 Value:',metrics.r2_score(y_train, DT.predict(X_train)) # Measuring how well training data fits

        # %matplotlib inline #plotting most important columns
        feature_importances = pd.Series(DT.feature_importances_, index=Predictors)
        feature_importances.nlargest(10).plot(kind='barh')

        pass 
'''
    
    

        
        
        
    

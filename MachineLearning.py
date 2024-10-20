import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
    
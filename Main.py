from Analysis import CSVReader
from MachineLearning import MachineLearning
from gui import GUI 


def main():
    csvReader = CSVReader()
    machineLearning = MachineLearning()
    gui = GUI()




    dataFrame = csvReader.readCsv('Cellphone.csv')
    print(dataFrame)
    csvReader.plotPriceDistribution(dataFrame)
    csvReader.visualizeScatterPlot(dataFrame['Price'], dataFrame['ppi'])
    cleanedData = csvReader.handleMissingValues(dataFrame)
    print(cleanedData)

    # Examine and interpret continuous predictors
    csvReader.selectFeaturesContinous(dataFrame, 'Price', 'cpu freq')

    # Analyse and visually represent category predictors
    csvReader.analyzeBoxPlot(dataFrame, 'Price', 'cpu core')
    csvReader.anova(dataFrame, 'Price', 'cpu core')

    finalFeatures = csvReader.features(dataFrame, 'Price', ['ppi', 'cpu freq', 'weight'], ['cpu core', 'internal mem'])
    print(f"Selected final features: {finalFeatures}")

    # Use one-hot encoding to transform the category columns into numerical representation.
    cateFeatures = ['cpu core', 'internal mem']
    dataFrameEncoded = csvReader.convertCategoricalToNumeric(dataFrame, cateFeatures)

    # Present the dataframe with values in numeric form.
    print(dataFrameEncoded)

    # call linear regression model
    machineLearning.LinearRegression('Cellphone.csv') 

    # Call SVM Regressor
    machineLearning.SVMRegressor('Cellphone.csv')

    # Call Random Forest
    machineLearning.RandomForest('Cellphone.csv')

    # Call Ada Boost
    machineLearning.AdaBoost('Cellphone.csv')

    # Call Decision Tree
    machineLearning.DecisionTree('Cellphone.csv')
    
    # Call the GUI
    # gui.loadGUI()
if __name__ == "__main__":
    main()
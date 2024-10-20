from Analysis import CSVReader
import nicegui
import gui

def main():
    csvReader = CSVReader()

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

if __name__ == "__main__":
    main()
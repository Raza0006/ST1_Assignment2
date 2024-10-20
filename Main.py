from Analysis import CSVReader
import nicegui
import gui

def main():
    csvReader = CSVReader()

    dataFrame = csvReader.readCsv('Cellphone.csv')
    print(dataFrame)
    csvReader.plotPriceDistribution(dataFrame)
    csvReader.visualizeScatterPlot(dataFrame['Price'], dataFrame['ppi'])
    cleaned_data = csvReader.handleMissingValues(dataFrame)
    print(cleaned_data)

    # Examine and interpret continuous predictors
    csvReader.selectFeaturesContinous(dataFrame, 'Price', 'cpu freq')

    # Analyse and visually represent category predictors
    csvReader.analyzeBoxPlot(dataFrame, 'Price', 'cpu core')
    csvReader.anova(dataFrame, 'Price', 'cpu core')

    final_features = csvReader.features(dataFrame, 'Price', ['ppi', 'cpu freq', 'weight'], ['cpu core', 'internal mem'])
    print(f"Selected final features: {final_features}")

    # Use one-hot encoding to transform the category columns into numerical representation.
    cate_features = ['cpu core', 'internal mem']
    df_encoded = csvReader.convert_categorical_to_numeric(dataFrame, cate_features)

    # Present the dataframe with values in numeric form.
    print(df_encoded)

if __name__ == "__main__":
    main()
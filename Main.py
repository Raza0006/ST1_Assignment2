from Analysis import CSVReader
import nicegui
import gui

def main():
    csvReader = CSVReader()

    dataFrame = csvReader.readCsv('Cellphone.csv')
    print(dataFrame)
    csvReader.plotPriceDistribution(dataFrame)
    csvReader.visualizeScatterPlot(dataFrame['Price'], dataFrame['ppi'])

if __name__ == "__main__":
    main()
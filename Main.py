from Analysis import CSVReader

def main():
    csvReader = CSVReader()
    dataFrame = csvReader.readCsv('Cellphone.csv')
    print(dataFrame)

    csvReader.plotPriceDistribution(dataFrame)

if __name__ == "__main__":
    main()
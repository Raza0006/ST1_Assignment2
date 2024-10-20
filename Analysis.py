# csv_reader.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class CSVReader:
    def __init__(self):
        # Set display options
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)

    def readCsv(self, filePath):
        try:
            # Read the csv file
            dataFrame = pd.read_csv(filePath)
            #remove duplicate rows
            dataFrame = dataFrame.drop_duplicates()
            #remove duplicate collumns and transpose the dataframe (.T)
            dataFrame = dataFrame.T.drop_duplicates().T

            return dataFrame
        except FileNotFoundError: # Raise an error if the file is not found
            print(f"File not found: {filePath}")
            return None
        except pd.errors.EmptyDataError: # Raise an error if the file is empty
            print(f"File is empty: {filePath}")
            return None
        except pd.errors.ParserError as e: # Raise an error if the file cannot be parsed
            print(f"Error parsing file: {filePath} - {e}")
            return None

    
    def plotPriceDistribution(self, dataFrame):
        # Plot a histogram to visualize the distribution of the price column
        plt.figure(figsize=(8, 6))
        sns.histplot(dataFrame['Price'], bins=50, kde=True)
        plt.title('Distribution of Price')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.show()
# csv_reader.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class CSVReader:
    def __init__(self):
        # Set display options
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)

    def readCsv(self, filePath): #Pasha's code
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

    
    def plotPriceDistribution(self, dataFrame): #Pasha's code
        # Plot a histogram to visualize the distribution of the price column
        plt.figure(figsize=(8, 6))
        sns.histplot(dataFrame['Price'], bins=50, kde=True)
        plt.title('Distribution of Price')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.show()

    def handleMissingValues(self, dataFrame): # MANASWINI PLEASE IMPLIMENT THIS FUNCTION
        # Handle missing values in the dataframe
        # dataFrame = dataFrame.dropna() # Drop rows with missing values
        '''
        YOU HAVE TO PICK ONE OF THESE OPTIONS TO HANDLE MISSING VALUES:
        Delete the missing value rows if there are only few records,
        Impute the missing values with MEDIAN value for continuous variables,
        Impute the missing values with MODE value for categorical variables,
        Interpolate the values based on nearby values,
        Interpolate the values based on business logic.
        '''
        
        return dataFrame
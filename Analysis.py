# csv_reader.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr


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

    def handleMissingValues(self, df):
        missing_values_threshold = 0.05 * len(df)
        df_dropped_rows = df.dropna(thresh=missing_values_threshold)

        # median values for continous values
        continuous_col = df_dropped_rows.select_dtypes(include=['float64', 'int64']).columns

        # replace missing values
        for col in continuous_col:
            df_dropped_rows[col].fillna(df_dropped_rows[col].median(), inplace=True)

        categorical_col = df_dropped_rows.select_dtypes(include=['object']).columns

        for col in categorical_col:
            df_dropped_rows[col].fillna(df_dropped_rows[col].mode()[0], inplace=True)

        for col in continuous_col:
            df_dropped_rows[col] = df_dropped_rows[col].interpolate(method='linear', inplace=False)

        def custom_price(df):
            # When price is missing, Calculating simlar items average price.
            if 'Price' in df.columns:
                df['Price'].fillna(df.groupby(['cpu core', 'ram'])['Price'].transform('mean'), inplace=True)
            return df
    
        # Missing values logic
        df_dropped_rows = custom_price(df_dropped_rows)
        
        return df_dropped_rows # Returning dropped rows
    
    def visualizeScatterPlot(self, target, predictor):
        plt.scatter(target, predictor)
        plt.xlabel('Target')
        plt.ylabel('Predictor')
        plt.title('Scatter Plot with Pearson\'s Correlation Value')
        correlationValue = target.corr(predictor)
        plt.text(0.5, -0.5, 'Pearson\'s Correlation: {:.2f}'.format(correlationValue), ha='center', va='center')
        plt.show()


    def selectFeaturesContinous(self, df, target, predictor):
        # Scatter plot
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x=predictor, y=target)
        plt.title(f'Scatter Plot: {predictor} vs {target}')
        plt.show()

        # Correlation
        correlation, _ = pearsonr(df[target], df[predictor])
        print(f'Correlation between {predictor} and {target}: {correlation}')        


    def analyzeBoxPlot(self, df, target, predictor):
        plt.figure(figsize=(8,6))
        sns.boxplot(data=df, x=predictor, y=target)
        plt.title(f'Box Plot:- {predictor} vs {target}')
        plt.show()


    def anova(self, df, target, predictor):
        # Box plot visualises the distribution of the constant variable for every grouping.
        plt.figure(figsize=(8,6))
        sns.boxplot(data=df, x=predictor, y=target)
        plt.title(f'Box Plot: {predictor} vs {target}')
        plt.show()

        # Sort the data according to the category prediction to get them ready for the ANOVA test.
        categories = df[predictor].unique()
        groups = [df[df[predictor] == category][target] for category in categories]

        # Performing ANOVA test
        f_stat, p_value = stats.f_oneway(*groups)

        # Results
        print(f'ANOVA Results: {predictor} vs {target}:')
        print(f'F-statistic: {f_stat}')
        print(f'P-value: {p_value}')

        # Interpretation of results
        if p_value < 0.05:
            print(f"Result: Relationship between {predictor} and {target} (Reject H0)")
        else:
            print(f"No significant result: No relationship between {predictor} and {target} (Fail to reject H0)")


    def features(self, df, target_var, continuous_fea, categorical_fea):
        sel_features = []
        
        # For continuous variables, look up the Pearson coefficient.
        for feature in continuous_fea:
            correlation, _ = pearsonr(df[target_var], df[feature])
            if abs(correlation) > 0.3:  # Select a criterion (e.g., > 0.3) to maintain significant correlations.

                sel_features.append(feature)
        
        # For categorical variables, the ANOVA test
        for feature in categorical_fea:
            categories = df[feature].unique()
            groups = [df[df[feature] == category][target_var] for category in categories]
            f_stat, p_value = stats.f_oneway(*groups)
            if p_value < 0.05:  # ANOVA results
                sel_features.append(feature)
        
        return sel_features
    

    def convert_categorical_to_numeric(self, df, categorical_fea):
        # Encoding to transform the category columns into numerical representation.
        df_enc = pd.get_dummies(df, columns=categorical_fea, drop_first=True)
        #Presenting the dataframe with values in numeric form.
        return df_enc
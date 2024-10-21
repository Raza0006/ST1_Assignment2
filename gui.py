'''
*******************************
Author: Raza
u123456 Assessment 1_Program1_(a) 22/ 11/2024
Programming:
*******************************

''' 

from Analysis import CSVReader
from MachineLearning import MachineLearning
import pandas as pd
from nicegui import ui
import matplotlib.pyplot as plt



class GUI:
    def __init__(self):
        pass

    def loadGUI(self, filePath):
        self.filePath = filePath
        csvReader = CSVReader()
        dataFrame = csvReader.readCsv('Cellphone.csv')
        cleanedData = csvReader.handleMissingValues(dataFrame)


        machineLearning = MachineLearning()  # Create an instance of MachineLearning

        def linearRegression():
            plt.close()
            machineLearning.LinearRegression(self.filePath)

        def decisionTree():
            plt.close()
            machineLearning.DecisionTree(self.filePath)

        def randomForest():
            plt.close()
            machineLearning.RandomForest(self.filePath)

        def svmRegressor():
            plt.close()
            machineLearning.SVMRegressor(self.filePath)
        def plotPriceDistribution():
            plt.close()
            csvReader.plotPriceDistribution(dataFrame)
        
        def visualiseScatterPlot():
            plt.close()
            csvReader.visualizeScatterPlot(dataFrame['Price'], dataFrame['ppi'])
        def selectFeaturesContinous():
            plt.close()
            csvReader.selectFeaturesContinous(dataFrame, 'Price', 'cpu freq')
        
        def analyzeBoxPlot():
            plt.close()
            csvReader.analyzeBoxPlot(dataFrame, 'Price', 'cpu core')
        def anova():
            plt.close()
            csvReader.anova(dataFrame, 'Price', 'cpu core')


        ui.html('<div style="background-color: #4CAF50; color: #fff; padding: 10px; text-align: center; width: 100%; margin: 0;">Mobile Phone Price Trend Analysis - ST1 Assessment 2</div>')
        with ui.row():
            with ui.column():
                ui.label("Analysis:")
                # Buttons for graphs before models
                ui.button("Distribution of Mobile Phone Prices", on_click= plotPriceDistribution)
                ui.button("Scatter Plot of Mobile Phone Prices", on_click= visualiseScatterPlot)
                ui.button("Select Features Continous", on_click= selectFeaturesContinous)
                ui.button("Analyze Box Plot", on_click= analyzeBoxPlot)
                ui.button("ANOVA", on_click= anova)

                # Buttons for algorithm models
                ui.label("Algorithms:")
                ui.button("Linear Regression", on_click=linearRegression)
                ui.button("Decision Tree", on_click=decisionTree)
                ui.button("Random Forest", on_click=randomForest)
                ui.button("SVM Regressor", on_click=svmRegressor)
            with ui.column():
                ui.label("Table of Data")
                with ui.card():
                    ui.markdown(cleanedData.to_markdown(index=False))

        ui.run()
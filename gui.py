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
        def Predict(value):
            plt.close()
            machineLearning.LinearRegressionNoGUI(self.filePath, value)
            ui.notify('Predicted Price: ' + str(machineLearning.LinearRegressionNoGUI(self.filePath, value)))

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
                value = ui.input(label='Pixels Per Inch', placeholder='start typing',
                        on_change=lambda e: result.set_text('you typed: ' + e.value),
                        validation={'Input too long': lambda value: len(value) < 20})
                result = ui.label()       
                ui.button("Predict", on_click= lambda:Predict(int(value.value)))         
                with ui.card():
                    ui.button("Linear Regression", on_click=linearRegression)

                    linear_regression_output = ui.markdown("""
                        The simplest, and most fundamental predictive model. 
                        It assumes a linear relationship between the dependent and independent variables. 
                        It is incredibly easy to interpret and utilises extrapolation to make predictive analysis. 
                        However, it sometimes fails to capture more complex data patterns.
                    """)
                linear_regression_output.style = "overflow-y: auto; height: 200px;"
                with ui.card():
                    ui.button("Decision Tree", on_click=decisionTree)
                    decision_tree_output = ui.markdown("""
                        These models use a tree-like structure to make predictions, hence the name. 
                        Each node represents a decision rule, while each leaf node corresponds to a final prediction. 
                        Although decision trees are easy to understand and implement, they are highly prone to overfitting. 
                        This means they may perform well on training data but often struggle to generalize effectively to new, unseen data. 
                        However, decision trees are incredibly important, in the sense that they are foundational to the implementation of other predictive models.
                    """)
                    decision_tree_output.style = "overflow-y: auto; height: 200px;"
                with ui.card():
                    ui.button("Random Forest", on_click=randomForest)
                    random_forest_output = ui.markdown("""
                        To solve the issues of overfitting, this model combines multiple simple decision trees into a more complicated structure. 
                        This increases redundancy and leads to a more accurate predictive model. 
                        However, this algorithm can prove to be slow and ineffective, especially if a large number of trees are used.
                    """)
                    random_forest_output.style = "overflow-y: auto; height: 200px;"
                with ui.card():
                    ui.button("SVM Regressor", on_click=svmRegressor)
                    svm_regressor_output = ui.markdown("""
                        Support Vector Machines works via the implementation of an optimal hyperplane that separates data points based on their classification. 
                        This method utilises kernel functions to handle high-dimensional data and works well with linear and non-linear data. 
                        It is especially useful in datasets with several classifications. 
                        However, this model requires extensive training, and the final model is often difficult to interpret.
                    """)
                    svm_regressor_output.style = "overflow-y: auto; height: 200px;"
            with ui.column():
                ui.label("Table of Data")
                with ui.card():
                    ui.markdown(cleanedData.to_markdown(index=False))

        ui.run()
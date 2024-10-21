'''
*******************************
Author: Raza, Manaswini, Rami, Lithika
u123456 Assessment 1_Program1_(a) 22/ 10/2024
Programming:
*******************************
''' 

from Analysis import CSVReader
from MachineLearning import MachineLearning
from gui import GUI 
from nicegui import ui

def main():
    
    ui.run()
    
    csvReader = CSVReader()
    machineLearning = MachineLearning()
    gui = GUI()

    dataFrame = csvReader.readCsv('Cellphone.csv')
    cleanedData = csvReader.handleMissingValues(dataFrame)

    finalFeatures = csvReader.features(dataFrame, 'Price', ['ppi', 'cpu freq', 'weight'], ['cpu core', 'internal mem'])

    # Use one-hot encoding to transform the category columns into numerical representation.
    cateFeatures = ['cpu core', 'internal mem']

    # Call the GUI
    gui.loadGUI('Cellphone.csv')

if __name__ in {"__main__", "__mp_main__"}:
    main()
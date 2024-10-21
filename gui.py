'''
*******************************
Author: Raza
u123456 Assessment 1_Program1_(a) 10/ 03/2024
Programming:
*******************************
''' 

from Analysis import CSVReader
import nicegui



class GUI:
    
    def __init__(self):
        pass
    def loadGUI(self): #Pasha's code
        ui = nicegui.App()
        csvReader = CSVReader()
        dataFrame = csvReader.readCsv('Cellphone.csv')
        table_data = dataFrame.values.tolist()
        table_columns = dataFrame.columns.tolist()
        table_data.insert(0, table_columns)
        ui.table(table_data) # THIS IS INCORRECT USE OF UI.TABL I WILL FIX THIS LATER
        ui.run()

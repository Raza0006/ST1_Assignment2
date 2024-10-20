from Analysis import CSVReader
import nicegui

class GUI:
    
    def __init__(self):
        pass
    def loadGUI(self): #Pasha's code
        app = nicegui.NiceGUI()
        csvReader = CSVReader()
        dataFrame = csvReader.readCsv('Cellphone.csv')
        app.table(dataFrame)
        app.run()

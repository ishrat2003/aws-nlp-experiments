from file.json import Json
from utility import Timer
from filesystem import Directory
import os

class Input:

  def __init__(self, filePath):
    file = Json()
    self.params = file.read(filePath)
    if not self.params:
      self.params = {}
    self.params['current_datetime'] = Timer.getFormatedDate()
    self.params['data_directory'] = self.getDataPath()
    self.params['output_directory'] = self.getOutputPath()
    return
  
  def getDataPath(self):
    return os.path.join(self.params['data_directory'], self.params['dataset_name'])
  
  def getOutputPath(self):
    directoryPath = os.path.join(self.params['output_directory'], self.params['dataset_name'], self.params['current_datetime'])
    outputDirectory = Directory(directoryPath)
    outputDirectory.create()
    return directoryPath

  def getAll(self):
    return self.params
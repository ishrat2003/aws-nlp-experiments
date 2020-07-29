from .core import Core as File
import os

class Base():
    
    def remove(self, filePath):
        file = File(filePath)
        return file.remove()
    
    def getFile(self, filename, writeHeader = True):
        path = os.path.join(self.path, filename)
        file = File(path, writeHeader)
        return file
    
    def getFilePath(self, fileName, path):
        return File.join(path, fileName)


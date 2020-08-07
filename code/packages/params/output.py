from file import Json

class Output(object):
    
    class __Output:
        def __init__(self):
            self.info = {}
            
        def addInfo(self, key, value):
            self.info[key] = value
            return
        
        def getInfo(self):
            return self.info
        
        def getInfoByKey(self, key):
            if key not in self.info.keys():
                return {}
            return self.info[key]
        
        def saveInfo(self, filePath):
            file = Json()
            return file.write(filePath, self.info)
    
    instance = None
    
    def __new__(cls):
        if not Output.instance:
            Output.instance = Output.__Output()
            
        return Output.instance
    
    def __getattr__(self, name):
        return getattr(self.instance, name)
    
    def __setattr__(self, name):
        return setattr(self.instance, name)
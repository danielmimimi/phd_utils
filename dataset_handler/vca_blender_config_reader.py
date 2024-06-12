import json

class VcaBlenderConfigReader:
    def __init__(self,config_path:str):
        self.config = config_path
        
    def load(self):
        with open(self.config,'r') as f:
            self.data = json.load(f) 
            
    def dump(self,path_to_write:str):
        with open(path_to_write, 'w') as file:
            json.dump(self.data, file)
            
    def get_data(self):
        return self.data

import torch
from torch.utils.data import Dataset

class LineData():
    
    def __init__(self):
        
        self.x = torch.tensor([[0.0],[1.0],[2.0],[3.0]])
        slef.y_true=torch.tensor([[1.0],[3.0],[5.0],[7.0]])

    def __len__(self):
        return len(self.x)
   
    def __getitem__(self, id):
        return (self.x[id], self.y_true[id])
   
    
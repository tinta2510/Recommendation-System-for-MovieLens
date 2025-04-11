from typing import Union
from math import sqrt

from torch.utils.data import Dataset
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

class RatingDataset(Dataset):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        ratings_df = pd.read_csv(file_path, usecols=range(3),
                                sep='\t', names=['user_id', 'item_id', 'rating'])
        self.user_tensor = torch.tensor(ratings_df['user_id'].values, dtype=torch.int32)
        self.item_tensor = torch.tensor(ratings_df['item_id'].values, dtype=torch.int32)
        self.rating_tensor = torch.tensor(ratings_df['rating'].values, dtype=torch.float32)
        
    def __getitem__(self, index: int) -> tuple[int, int, float]:
        return self.user_tensor[index], self.item_tensor[index], self.rating_tensor[index]
    
    def __len__(self) -> int:
        return len(self.user_tensor)
    
def train(model: nn.Module, 
          dataloader: DataLoader, 
          criterion: nn.Module, 
          optimizer: torch.optim.Optimizer, 
          device: Union[torch.device, str],
          epochs: int = 2):
    model.train()
    for i in range(epochs):
        for users, items, ratings in dataloader: 
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            # Zero out the grad
            optimizer.zero_grad()
            # Predict the ratings
            outputs = model(users, items)
            # Calculate loss
            loss = criterion(outputs, ratings)
            # Calculate gradient descent
            loss.backward()
            # Update weight
            optimizer.step()
        
        print(f"Loss and RMSE after {i} epochs: ", evaluate(model, dataloader, criterion, device))

def evaluate(model: nn.Module, 
               dataloader: DataLoader, 
               criterion: nn.Module, 
               device: Union[torch.device, str]):     
        model.eval()
        total_loss = 0
        error = 0
        total_ratings = 0
        with torch.no_grad():
            for users, items, ratings in dataloader:
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                outputs = model(users, items)
                loss = criterion(outputs, ratings)
                total_loss += loss
                
                error += ((outputs-ratings)**2).sum().item()
                total_ratings += ratings.size(0)
        
        return total_loss / len(dataloader), sqrt(error/total_ratings)        
    
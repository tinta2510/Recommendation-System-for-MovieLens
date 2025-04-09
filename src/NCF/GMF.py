import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import RatingDataset, train, evaluation
from torch.utils.data import DataLoader
learning_rate = 0.001
n_factors = 32

class GMF(nn.Module):
    """
    
    Note: Convert input user_id and item_id to 0-based indices
    """
    def __init__(self, 
                 n_users: int, 
                 n_items: int, 
                 n_factors: bool, 
                 for_NeuMF: bool = False):
        super().__init__()
        self.for_NeuMF = for_NeuMF
        self.user_embed = nn.Embedding(n_users, n_factors)
        self.item_embed = nn.Embedding(n_items, n_factors)
        if not self.for_NeuMF:
            self.linear = nn.Linear(n_factors, 1)
            self.sigmoid = nn.Sigmoid()
        
        self._init_weight()
        
    def _init_weight(self):
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        if not self.for_NeuMF:
            nn.init.xavier_normal_(self.linear.weight, gain=nn.init.calculate_gain('sigmoid'))
    
    def forward(self, users, items):
        users = users - 1
        items = items - 1
        user_embeds = self.user_embed(users)
        item_embeds = self.item_embed(items)
        out = user_embeds * item_embeds
        if not self.for_NeuMF:
            out = self.linear(out)
            out = (self.sigmoid(out)*5).view(-1)
        return out

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(1, 6):
        training_filepath = f'./data/ml-100k/u{i}.base'
        testing_filepath = f'./data/ml-100k/u{i}.test'
        
        training_dataset = RatingDataset(training_filepath)
        testing_dataset = RatingDataset(testing_filepath)
        
        training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
        testing_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=False)
        
        model = GMF(943, 1682, n_factors).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train(model, training_dataloader, criterion, optimizer, device)
        loss, rmse = evaluation(model, training_dataloader, criterion, device)
        print(f"Dataset {i} - Training: Loss: {loss}, RMSE: {rmse}")
        loss, rmse = evaluation(model, testing_dataloader, criterion, device)
        print(f"Dataset {i} - Testing: Loss: {loss}, RMSE: {rmse}")
        
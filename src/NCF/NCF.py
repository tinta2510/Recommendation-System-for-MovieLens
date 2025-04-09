from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import RatingDataset, train, evaluation
from torch.utils.data import DataLoader

from GMF import GMF
from MLP import MLP

class NeuralCF(nn.Module):
    def __init__(self, 
                 n_users: int,
                 n_items: int,
                 n_factors_GMF: Optional[int] = None,
                 hidden_layers_MLP: Optional[list[int]] = None,
                 GMF_model: Optional[GMF] = None,
                 MLP_model: Optional[MLP] = None):
        assert not (GMF_model is None and n_factors_GMF is None), \
            "GMF_model and n_factors_GMF cannot both be None"
        assert not (MLP_model is None and hidden_layers_MLP is None), \
            "MLP_model and hidden_layers_MLP cannot both be None"
        super().__init__()
        if GMF_model:
            GMF_model.for_NeuMF = True
        if MLP_model:
            MLP_model.for_NeuMF = True
        self.GMF = (GMF_model 
                        if GMF_model 
                        else GMF(n_users, n_items, n_factors_GMF, for_NeuMF=True))
        self.MLP = (MLP_model 
                        if MLP_model 
                        else MLP(n_users, n_items, hidden_layers_MLP, for_NeuMF=True))
        
        self.relu_GMF = nn.ReLU()
        self.relu_MLP = nn.ReLU()
        
        self.neuMF_layer = nn.Linear(n_factors_GMF + hidden_layers_MLP[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
        self._init_weight()
        
    def _init_weight(self):
        nn.init.xavier_uniform_(self.neuMF_layer.weight, gain=nn.init.calculate_gain('sigmoid'))
    
    def forward(self, users, items):
        # Already convert to 0-based index in GMF and MLP
        gmf_out = self.GMF(users, items)
        mlp_out = self.MLP(users, items)
        concat_out = torch.concat([gmf_out, mlp_out], dim=1)
        out = self.neuMF_layer(concat_out)
        out = (self.sigmoid(out)*5).view(-1)
        return out
        
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.001
    for i in range(1, 6):
        training_filepath = f'./data/ml-100k/u{i}.base'
        testing_filepath = f'./data/ml-100k/u{i}.test'
        
        training_dataset = RatingDataset(training_filepath)
        testing_dataset = RatingDataset(testing_filepath)
        
        training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
        testing_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=False)
        
        model = NeuralCF(943, 1682, 8, [32, 16, 8]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train(model, training_dataloader, criterion, optimizer, device)
        loss, rmse = evaluation(model, testing_dataloader, criterion, device)
        print(f"Dataset {i}: Loss: {loss}, RMSE: {rmse}")
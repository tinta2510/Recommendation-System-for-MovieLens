from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

from NCF.GMF import GMF
from NCF.MLP import MLP
from NCF.utils import RatingDataset, train, evaluate

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
        
        self.neuMF_layer = nn.Linear(self.GMF.n_factors + self.MLP.hidden_layers[-1], 1)
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
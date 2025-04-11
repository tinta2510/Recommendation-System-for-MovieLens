import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

from NCF.utils import RatingDataset, train, evaluate

learning_rate = 0.001

class MLP(nn.Module):
    def __init__(self, 
                 n_users: int, 
                 n_items: int, 
                 hidden_layers: list[int],
                 for_NeuMF: bool = False):
        super().__init__()
        self.for_NeuMF = for_NeuMF
        self.user_embed = nn.Embedding(n_users, hidden_layers[0]//2)
        self.item_embed = nn.Embedding(n_items, hidden_layers[0]//2)
        
        mlp = []
        for i in range(len(hidden_layers)-1):
            mlp.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            mlp.append(nn.ReLU())
        
        self.sequential = nn.Sequential(*mlp)
        if not self.for_NeuMF: 
            self.predict_layer = nn.Linear(hidden_layers[-1], 1)
            self.sigmoid = nn.Sigmoid()
        
        self._init_weight()
        
    def _init_weight(self):
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        for layer in self.sequential:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if not self.for_NeuMF: 
            nn.init.xavier_uniform_(self.predict_layer.weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, users, items):
        # Apply 0-based indexing conversion
        users = users - 1
        items = items - 1
        
        user_embed = self.user_embed(users) # [batch_size, n_features]
        item_embed = self.item_embed(items) # [batch_size, n_features]
        concat_embed = torch.cat([user_embed, item_embed], dim=1)
        output = self.sequential(concat_embed)
        if not self.for_NeuMF:
            output = self.predict_layer(output)
            output = (self.sigmoid(output)*5).view(-1)
        return output

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(1, 6):
        training_filepath = f'./data/ml-100k/u{i}.base'
        testing_filepath = f'./data/ml-100k/u{i}.test'
        
        training_dataset = RatingDataset(training_filepath)
        testing_dataset = RatingDataset(testing_filepath)
        
        training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
        testing_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=False)
        
        model = MLP(943, 1682, [128, 64, 32, 16, 8])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train(model, training_dataloader, criterion, optimizer, device)
        loss, rmse = evaluate(model, testing_dataloader, criterion, device)
        print(f"Dataset {i}: Loss: {loss}, RMSE: {rmse}")
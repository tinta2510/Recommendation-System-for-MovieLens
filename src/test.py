import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import math
from scipy.sparse import csr_matrix
class NeighborhoodCF:
    def __init__(self, k: int = 10, distance_func = cosine_distances, uuCF: bool = True):
        """
        Initialize the NeighborhoodCF class.
        
        Args:
            user_item_matrix (DataFrame): Matrix with users as rows, 
                items as columns and ratings as values.
            k (int): Number of neighbors to use for prediction.
            distance_func (function): Function to compute distance between vectors.
            uuCF (bool): If True, use user-user collaborative filtering, 
                else item-item.
        """
        self.uuCF = uuCF
        self.distance_func = distance_func
        self.k = k
    
    def fit(self, ratings):
        self.ratings = ratings if self.uuCF else ratings[:, [1, 0, 2]]
        # number of users/items equals max id because id start from 0
        self.n_users = int(np.max(self.ratings[:, 0])) 
        self.n_items = int(np.max(self.ratings[:, 1])) 
        self._normalize_ratings()
        self._compute_similarity()
        
    def _normalize_ratings(self):
        """
        Normalize the ratings by subtracting the mean rating for each user.
        """
        self.average_ratings = np.zeros((self.n_users,))
        self.normalized_ratings = self.ratings.astype(float).copy()
        # REMEMBER: ID starting from 1
        for i in range(self.n_users):
            # Idx of ratings of user with id i+1
            ids = np.where(self.ratings[:,0] == i+1)
            
            # Average ratings of user with id i+1
            self.average_ratings[i] = np.mean(self.ratings[ids, 2]) if len(ids[0]) == 0 else 0
            
            # Normalize ratings table
            self.normalized_ratings[ids, 2] -= self.average_ratings[i]
        
        # User-item rating matrix
        self.u_i_matrix = csr_matrix((self.normalized_ratings[:, 2], # values
                                        (self.normalized_ratings[:, 0] - 1,   # row indices (subtract 1 for 0-based indexing)
                                        self.normalized_ratings[:, 1] - 1)    # column indices (subtract 1 for 0-based indexing)
                                    ),      
                                    shape=(self.n_users, self.n_items))     # matrix dimensions
        
    def _compute_similarity(self):
        self.similarity_matrix = self.distance_func(self.u_i_matrix, self.u_i_matrix)
        
    def __pred(self, u_id, i_id):
        # Scale u_id and i_id starting from 0
        u_id -= 1
        i_id -= 1
        
        # Step 1: Find users who rated item i
        item_mask = self.ratings[:, 1] == i_id + 1
        if not np.any(item_mask):
            return self.average_ratings[u_id]
            
        u_ids = self.ratings[item_mask, 0].astype(int) - 1 # This is user_id (start from 1)!!!

        # Step 2: Find most similar users among u_ids
        nearest_uids = np.argsort(self.similarity_matrix[u_id, u_ids])[-self.k:]
        

        ratings = (np.sum(self.normalized_ratings[nearest_uids][:, 2]
                          * self.similarity_matrix[u_id, nearest_uids]) 
                   / (np.sum(self.similarity_matrix[u_id, nearest_uids]) + 1e-8))
        return ratings + self.average_ratings[u_id]
    
    def predict(self, u_id, i_id):
        if self.uuCF:
            return self.__pred(u_id, i_id)
        return self.__pred(i_id, u_id)
    

    def evaluate(self, test_ratings: np.ndarray):
        self.test_ratings = test_ratings if self.uuCF else test_ratings[:, [1, 0, 2]]
        mse = 0
        for u, i, rate in test_ratings:
            mse += (rate - self.predict(u, i)) ** 2
        return math.sqrt(mse/len(self.test_ratings))
    
for i in range(1, 6):
    train_ratings = pd.read_csv(f'./data/ml-100k/u{i}.base', usecols=range(3),
                                sep='\t', names=['user_id', 'item_id', 'rating'])
    test_ratings = pd.read_csv(f'./data/ml-100k/u{i}.test', usecols=range(3),
                                sep='\t', names=['user_id', 'item_id', 'rating'])
    recommender = NeighborhoodCF(k=20, uuCF=False)
    recommender.fit(train_ratings.values)
    print(f"Evaluation of model on dataset {i}: ")
    print(recommender.evaluate(train_ratings.values))
    print(recommender.evaluate(test_ratings.values))


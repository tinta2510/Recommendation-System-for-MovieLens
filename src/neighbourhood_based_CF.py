import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import math
from scipy.sparse import csr_matrix

class NeighborhoodCF:
    def __init__(self, n_users: int, n_items: int, k: int = 10, 
                 distance_func = cosine_distances, uuCF: bool = True):
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
        if self.uuCF:
            self.n_users = n_users
            self.n_items = n_items
        else:
            self.n_users = n_items
            self.n_items = n_users
        self.distance_func = distance_func
        self.k = k
        
    def fit(self, ratings):
        self.ratings = ratings if self.uuCF else ratings[:, [1, 0, 2]]
        # number of users/items equals max id because id start from 0
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
            self.average_ratings[i] = np.mean(self.ratings[ids, 2]) if len(ids[0]) != 0 else 0
            
            # Normalize ratings table
            self.normalized_ratings[ids, 2] -= self.average_ratings[i]
        
        # User-item rating matrix
        self.u_i_matrix = csr_matrix((self.normalized_ratings[:, 2], # values
                                        (self.normalized_ratings[:, 0].astype(np.int32) - 1,   # row indices (subtract 1 for 0-based indexing)
                                        self.normalized_ratings[:, 1].astype(np.int32) - 1)    # column indices (subtract 1 for 0-based indexing)
                                    ),      
                                    shape=(self.n_users, self.n_items))     # matrix dimensions
        
    def _compute_similarity(self):
        self.similarity_matrix = self.distance_func(self.u_i_matrix, self.u_i_matrix)
        
    def __pred(self, u_id, i_id):
        # Step 1: Find users who rated item i
        item_mask = self.ratings[:, 1] == i_id
        if not np.any(item_mask):
            return self.average_ratings[u_id-1]
            
        u_ids = self.ratings[item_mask, 0].astype(int) 

        # Step 2: Find most similar users among u_ids. user_id (start from 1)!!!
        nearest_uids = np.argsort(self.similarity_matrix[u_id-1, u_ids-1])[-self.k:]
        
        ratings = (np.sum(self.normalized_ratings[nearest_uids-1][:, 2]
                          * self.similarity_matrix[u_id-1, nearest_uids-1]) 
                   / (np.sum(self.similarity_matrix[u_id-1, nearest_uids-1]) + 1e-8))
        return ratings + self.average_ratings[u_id-1]
    
    def predict(self, u_id, i_id):
        if self.uuCF:
            return self.__pred(u_id, i_id)
        return self.__pred(i_id, u_id)
    

    def evaluateRMSE(self, test_ratings: np.ndarray):
        self.test_ratings = test_ratings if self.uuCF else test_ratings[:, [1, 0, 2]]
        mse = 0
        for u, i, rate in test_ratings:
            mse += (rate - self.predict(u, i)) ** 2
        return math.sqrt(mse/len(self.test_ratings))
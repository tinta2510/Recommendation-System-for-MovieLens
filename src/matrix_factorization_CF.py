import numpy as np
import pandas as pd
class MatrixFactorizationCF:
    def __init__(self, k = 10, learning_rate=0.01, alpha  = 0.01):  
        """
        Args:
            k (int): number of latent vectors
            learning_rate (float):
        """
        self.k = k
        self.learning_rate = learning_rate
        self.alpha = alpha
        
    def fit(self, ratings, max_iter=10000):
        """
        Args:
            ratings (np.ndarray): column 0th (user), column 1st (item), column 2nd (rating)
        """
        self.ratings = ratings
       
        self.n_users = int(np.max(self.ratings[:, 0])) # User_id starts from 1
        self.n_items = int(np.max(self.ratings[:, 1])) # Item_id starts from 1
        self.n_ratings = len(ratings)
        
        self._normalize_ratings()

        # User feature matrix M*K
        self.U = np.random.randn(self.n_users, self.k) 
        # Item feature matrix K*N
        self.I = np.random.randn(self.k, self.n_items)
        
        for i in range(max_iter):
            self._update_U()
            self._update_I()
            if i  % 50 == 0:
                print(f"RMSE after {i} iters: ", self.evaluate(ratings), "Loss: ", self.loss(ratings))
            
    def predict(self, u_id, i_id):
        normalized_rating = self.U[u_id-1].dot(self.I[:, i_id-1])
        # TODO if user_based
        return np.clip(normalized_rating + self.average_ratings[u_id-1], 0, 5)
    
    def loss(self, ratings):
        L = 0
        for u_id, i_id, rating in ratings:
            L += (rating - self.U[u_id-1,:].dot(self.I[:, i_id-1]))**2
        loss = L/(2*len(ratings)) + self.alpha/2*(np.linalg.norm(self.U) + np.linalg.norm(self.I))
        return loss

    def evaluate(self, ratings):
        mse = 0
        for u_id, i_id, rating in ratings:
            mse += (rating - self.predict(u_id, i_id))**2
        return np.sqrt(mse/len(ratings))
        
    def _get_item_rated_by_user(self, u_id):
        rating_indices = np.where(self.normalized_ratings[:, 0] == u_id)[0].astype(np.int32)
        item_ids = self.normalized_ratings[rating_indices, 1].astype(np.int32)
        ratings = self.normalized_ratings[rating_indices, 2]
        return (item_ids, ratings)
    
    def _get_user_rating_item(self, i_id):
        rating_indices = np.where(self.normalized_ratings[:, 1] == i_id)[0].astype(np.int32)
        user_ids = self.normalized_ratings[rating_indices, 0].astype(np.int32)
        ratings = self.normalized_ratings[rating_indices, 2]
        return (user_ids, ratings)
    
    def _normalize_ratings(self):
        # TODO
        # if user_based: ?
        
        self.average_ratings = np.zeros((self.n_users,))
        self.normalized_ratings = self.ratings.astype(float).copy()
        for i in range(self.n_users):
            # Indices of ratings by user with id i+1
            indices = np.where(self.ratings[:, 0] == i+1)
            self.average_ratings[i] = np.mean(self.ratings[indices, 2]) if len(indices) != 0 else 0
            # Subtract ratings of user i+1 by average_ratings
            self.normalized_ratings[indices, 2] -= self.average_ratings[i]
            
    def _update_U(self):
        for m in range(self.n_users):
            item_ids, ratings = self._get_item_rated_by_user(m+1)
            # current predicted ratings
            ratings_m_hat = ratings - self.U[m].dot(self.I[:, item_ids-1])
            grad_um = ((ratings_m_hat).dot(self.I[:, item_ids-1].T)/self.n_ratings
                        + self.alpha*self.U[m])
            self.U[m] -= self.learning_rate*grad_um
    
    def _update_I(self):
        for n in range(self.n_items):
            user_ids, ratings = self._get_user_rating_item(n)
            ratings_n_hat = ratings - self.U[user_ids-1,:].dot(self.I[:,n])
            grad_in = (self.U[user_ids-1, :].T.dot(ratings_n_hat)/self.n_ratings 
                       + self.alpha*self.I[:, n])
            self.I[:, n] -= self.learning_rate*grad_in
        
for i in range(1, 6):
    train_ratings = pd.read_csv(f'./data/ml-100k/u{i}.base', usecols=range(3),
                                sep='\t', names=['user_id', 'item_id', 'rating'])
    test_ratings = pd.read_csv(f'./data/ml-100k/u{i}.test', usecols=range(3),
                                sep='\t', names=['user_id', 'item_id', 'rating'])
    recommender = MatrixFactorizationCF(10, alpha=0.1, learning_rate=0.5)
    recommender.fit(train_ratings.values, max_iter=100)
    print(f"Evaluation of model on dataset {i}: ")
    print("RMSE on training data: ", recommender.evaluate(train_ratings.values))
    print("RMSE on testing data: ", recommender.evaluate(test_ratings.values))
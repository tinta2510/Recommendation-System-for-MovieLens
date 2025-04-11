import numpy as np
from numpy import ndarray
from sklearn.linear_model import Ridge
from math import sqrt
class ContentBasedFiltering:
    def __init__(self, n_users: int, n_items: int, item_features: ndarray, alpha=0.01):
        """
        
        Args:
            n_users (int): Number of users.
            n_items (int): Number of items.
            item_features (ndarray): Array of item features.
                Shape (n_items, n_features).
            alpha (float): Regularization parameter for Ridge regression.
                Default is 0.01.
        """
        self.n_users = n_users
        self.n_items = n_items
        self.item_features = item_features
        self.alpha = alpha
        self.n_features = item_features.shape[1]
        self.W = None
        self.b = None
        
    def fit(self, training_ratings: ndarray):
        """
        Fit the model using training ratings.
        Args:
            training_ratings (ndarray): ratings data.
                Shape (n_ratings, 3) where each row contains user_id, 
                item_id, and rating.
        """
        self.W = np.zeros((self.n_users, self.n_features))
        self.b = np.zeros((self.n_users, 1))
        for user_id in range(1, self.n_users+1):
            # Get ratings of the current user
            user_ratings = training_ratings[training_ratings[:, 0] == user_id, :]
            
            model = Ridge(alpha=self.alpha, fit_intercept=True)
            # Note that item_id is 1-indexed
            rated_item_ids = user_ratings[:, 1]
            model.fit(self.item_features[rated_item_ids-1, :], user_ratings[:, 2])

            self.W[user_id-1] = model.coef_
            self.b[user_id-1] = model.intercept_
        
    def predict(self, user_id: int, item_id: int):
        if self.W is None:
            raise ValueError("Model has not been fitted yet.")
        if user_id < 1 or user_id > self.n_users:
            raise ValueError("User ID out of range.")
        if item_id < 1 or item_id > self.n_items:
            raise ValueError("Item ID out of range.")
        return np.dot(self.W[user_id-1], self.item_features[item_id-1, :]) + self.b[user_id-1]
    
    def evaluateRMSE(self, test_ratings: ndarray):
        if self.W is None:
            raise ValueError("Model has not been fitted yet.")
        sum_squared_error = 0
        for user_id, item_id, rating in test_ratings:
            sum_squared_error += (rating - self.predict(user_id, item_id)) ** 2
        mse = sum_squared_error / len(test_ratings)
        return sqrt(mse)
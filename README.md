# Recommendation System on MovieLens dataset
## Project Overview
This project builds some classic and modern recommendation algorithms from scratch. The main objective is to have a better understanding of how different recommendation systems work and to compare their performance on the [MovieLens dataset](https://grouplens.org/datasets/movielens/).

The implemented methods include:
- Content-Based Filtering: Recommends items similar to those the user liked in the past based on item features.
- Neighborhood-Based Collaborative Filtering: Predicts user preferences by looking at similar users (user-user) or similar items (item-item).
- Matrix Factorization: Decomposes the user-item interaction matrix into lower-dimensional matrices to capture latent features.
- Neural Collaborative Filtering (NCF): Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) for a more expressive model, which can learn complex, non-linear user-item relations. [<a href="#ref1">1</a>]

## Dataset 
This project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/) 100k dataset, a benchmark dataset widely used in academic research for evaluating recommender systems.
- Contains 100,000 ratings (1–5 stars) from 943 users on 1682 movies.
- Data is split into five folds using the provided `u1.base`, `u1.test`, ..., `u5.base`, `u5.test` format for cross-validation.

All models in this project are evaluated across these five splits for consistency and fair comparison.

### Objective
The aim of recommendation systems is to predict the rating a user would give to an item they haven't rated yet. 

### Data Exploration
Data has been loaded into dataframes using pandas. It had three main columns: user_id, item_id, and rating. It had been analyezed and visulaized to gain insights before building the models. The details can be seen in [src/data_exploration.ipynb](src/data_exploration.ipynb). The following are some of the key observations:

- The distribution of ratings is not uniform, with a higher concentration of ratings around the middle values (3-4).
![Rating distribution](images/rating-distribution.png)

## References
<span id="ref1">[1]</span> Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural Collaborative Filtering. In Proceedings of the 26th International Conference on World Wide Web (WWW '17). International World Wide Web Conferences Steering Committee, Republic and Canton of Geneva, CHE, 173–182. https://doi.org/10.1145/3038912.3052569




import argparse
import json
import sys
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors as NN
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)


amount_restrictions = 6
amount_people = 491
amount_genres = 23


def new_features_similarity(movie1_row, movie2_row):
    genres_coef = 0.5
    people_coef = 0.3
    critics_coef = audience_coef = (1 - genres_coef - people_coef) / 2
    restriction_coef = 0

    critics_score = 1 - abs(movie1_row[0] - movie2_row[0]) / 100
    audience_score = 1 - abs(movie1_row[1] - movie2_row[1]) / 100
    # try:
    #     genres_score = jaccard_score(movie1_row[2:2 + amount_genres], movie2_row[2:2 + amount_genres])
    #     print(movie1_row[2:2 + amount_genres], movie2_row[2:2 + amount_genres])
    #     print('genres score', genres_score)
    # except:
    #     print(movie1_row[2:2 + amount_genres], movie2_row[2:2 + amount_genres])
    #     raise
    # people_score = jaccard_score(movie1_row[2 + amount_genres:2 + amount_genres + amount_people], movie2_row[2 + amount_genres:2 + amount_genres + amount_people])
    # restriction_score = jaccard_score(movie1_row[-amount_restrictions:], movie2_row[-amount_restrictions:])
    genres_score = cosine_similarity(movie1_row[2:2 + amount_genres].reshape(1, -1), movie2_row[2:2 + amount_genres].reshape(1, -1))
    people_score = cosine_similarity(movie1_row[2 + amount_genres:2 + amount_genres + amount_people].reshape(1, -1), movie2_row[2 + amount_genres:2 + amount_genres + amount_people].reshape(1, -1))
    restriction_score = cosine_similarity(movie1_row[-amount_restrictions:].reshape(1, -1), movie2_row[-amount_restrictions:].reshape(1, -1))
    return critics_score * critics_coef + audience_score * audience_coef + genres_score * genres_coef + people_score * people_coef + restriction_coef * restriction_score
    # return 1 - (critics_score * critics_coef + audience_score * audience_coef + genres_score * genres_coef + people_score * people_coef + restriction_coef * restriction_score)


def custom_similarity(movie1_row, movie2_row):
    # print(movie1_row.shape, movie2_row.shape)
    # print(movie1_row)
    # print(movie2_row)
    # print(amount_critics, movie1_row[amount_critics:].shape, movie2_row[amount_critics:].shape)
    alpha = 0
    correlation = 1 - pearsonr(movie1_row[:amount_critics], movie2_row[:amount_critics])[0]
    custom_sim = new_features_similarity(movie1_row[amount_critics:], movie2_row[amount_critics:])
    return alpha * correlation + (1 - alpha) * custom_sim


class OurCF:

    def __init__(self, shrink_term=1):
        self.shrink_term = shrink_term
        self.dataset = None
        self.NN = NN(metric=custom_similarity)

    def fit(self, df):
        '''
            takes in pandas dataframe with columns ['critic_id', 'movie_id', 'score']
            constructs the NN model with correlation distance
        '''

        self.dataset = df
        self.NN.fit(self.dataset)

    def get_score(self, user_id, movie_id, neighbors):
        '''
            Getting a score for a given user for a given movie, using the similarity matrix

            using a formula from surprise package (https://surprise.readthedocs.io/en/stable/knn_inspired.html, KNNWithMeans):

                            sum_{v in people} (sim(u, v) * (r_{vi} - mu_v))
            r_{ui} = mu_u + -----------------------------------------------   for user-to-user, and
                                    c + sum_{v in people} sim(u, v)


                            sum_{j in items} (sim(i, j) * (r_{uj} - mu_j))
            r_{ui} = mu_i + -----------------------------------------------   for item-to-item recommendations
                                    c + sum_{j in items} sim(i, j)

            where mu_i, mu_u, mu_j, and mu_v are mean ratings for particular item/user
        '''
        numerator = 0
        denominator = self.shrink_term
        subdf = self.dataset.loc[movie_id]  # getting all critics who left score for that movie
        working_series = self.dataset[user_id]
        item_neighbors = self.NN.kneighbors([self.dataset.loc[movie_id]], n_neighbors=11)
        item_neighbors = list(self.dataset.iloc[item_neighbors[1][0][1:]].index)
        for j in item_neighbors:
            j_series = self.dataset.loc[j]
            mu_j = j_series[j_series != 0].mean()
            r_uj = working_series[j]
            if r_uj == 0:
                continue
            sim_ij = custom_similarity(j_series, subdf)
            numerator += sim_ij * (r_uj - mu_j)
            denominator += sim_ij

        mu = subdf[subdf != 0].mean()  # mu_u/mu_i
        return mu + numerator / denominator

    def recommend(self, user_id, k):
        """ recommendation function: gets scores for all (or closest to top-ranked by user) movies and ranks them, giving out top k of them """
        scored_list = []
        print('Ranking movies')
        movies_pool = set()
        user_df = self.dataset[user_id]
        top_ranked_movies = user_df[user_df == user_df.max()].index
        for movie in top_ranked_movies:
            movie_neighbors = self.NN.kneighbors([self.dataset.loc[movie]], n_neighbors=3)  # 3 is a magic number
            movies_pool.update(set(self.dataset.iloc[movie_neighbors[1][0][1:]].index))
        neighbors = 'all'
        for movie_id in tqdm(movies_pool):
            scored_list.append((movie_id, self.get_score(user_id, movie_id, neighbors)))
        scored_list.sort(key=lambda x: x[1], reverse=True)
        return scored_list[:k]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--amount-recommendations', '-k', default=10)
    parser.add_argument('--clean', '-c', default='not_clean', choices=['clean', 'not_clean'])
    args = parser.parse_args()

    model = OurCF()

    print('\nLoading data')
    df_reviews = pd.read_csv('reviews_clean.tsv', sep='\t')[['movie_id', 'critic_id', 'score']].drop_duplicates(subset=['movie_id', 'critic_id'])
    features = json.load(open('films_features.json'))
    one_hot_encoded_features = pd.read_csv('film_features.tsv', sep='\t').set_index('movie_id', drop=True)

    popularity_thres = 50
    df_new = df_reviews.copy()

    if args.clean == 'clean':
        print('\tinitial shape:', df_reviews.shape)
        df_reviews = df_reviews[df_reviews['movie_id'].isin(features)]
        prev_shape = df_reviews.shape
        new_shape = 0
        df_med = df_reviews

        # removing critics and movies with less than 50 reviews (until convergence)
        while prev_shape != new_shape:
            prev_shape = df_med.shape
            df_movies_cnt = pd.DataFrame(df_med.groupby('movie_id').size(), columns=['count'])
            popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
            df_ratings_drop_movies = df_med[df_med.movie_id.isin(popular_movies)]
            df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('critic_id').size(), columns=['count'])
            prolific_users = list(set(df_users_cnt.query('count >= @popularity_thres').index))
            df_med = df_ratings_drop_movies[df_ratings_drop_movies.critic_id.isin(prolific_users)]
            new_shape = df_med.shape

        df_new = df_med
        print('\tnew shape:', df_new.shape)

        df_new.to_csv('reviews_clean.tsv', index=False, sep='\t')

    print('\tdistinct movies:', df_new['movie_id'].nunique())
    print('\tdistinct critics:', df_new['critic_id'].nunique())

    # this is a test user to check validity of the model
    me = pd.DataFrame([{'movie_id': 'iron_man', 'critic_id': 'test_user', 'score': 5},
                       {'movie_id': 'captain_america_civil_war', 'critic_id': 'test_user', 'score': 5},
                       {'movie_id': 'captain_america_the_winter_soldier_2014', 'critic_id': 'test_user', 'score': 5},
                       {'movie_id': 'captain_america_the_first_avenger', 'critic_id': 'test_user', 'score': 5},
                       {'movie_id': 'avengers_age_of_ultron', 'critic_id': 'test_user', 'score': 5},
                       {'movie_id': 'avengers_infinity_war', 'critic_id': 'test_user', 'score': 5},
                       {'movie_id': 'avengers_endgame', 'critic_id': 'test_user', 'score': 5}])

    df_new = df_new.append(me).reset_index(drop=True).pivot(index='movie_id', columns='critic_id', values='score').fillna(0)
    amount_critics = df_new.shape[1]
    final_df = pd.merge(df_new, one_hot_encoded_features, left_index=True, right_index=True).astype(int)
    print('Data loaded\n')

    # print(custom_similarity(final_df.loc['trollhunter'].to_numpy(), final_df.loc['avengers_endgame'].to_numpy()))
    # print(custom_similarity(final_df.loc['0814255'].to_numpy(), final_df.loc['avengers_endgame'].to_numpy()))
    model.fit(final_df)

    critic = 'test_user'
    recommendations = enumerate(model.recommend(critic, args.amount_recommendations))
    print(f'\nHere are top-{args.amount_recommendations} recommendations for critic {critic}:')
    for ind, tple in recommendations:
        print(f'\tMovie {ind+1}: {tple[0]} w score {tple[1]}')

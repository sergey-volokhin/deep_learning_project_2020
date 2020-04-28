import json
from collections import Counter
import pandas as pd

people_frequency = []
film_features = json.load(open('films_features.json', 'r'))
ratings = set([features['rating'] for features in film_features.values()])

genres = set(sum([features['genre'] for features in film_features.values()], []))
most_common_people = [x[0] for x in Counter(sum([features['people'] for features in film_features.values()], [])).most_common(491)]
new_dicts = []
for film, features in film_features.items():
    new_dict = {key: features[key] for key in ['critic_score', 'audience_score', 'rating']}
    new_dict['movie_id'] = film
    for person in most_common_people:
        if person in features['people']:
            new_dict[person] = 1
        else:
            new_dict[person] = 0
    for genre in genres:
        if genre in features['genre']:
            new_dict[genre] = 1
        else:
            new_dict[genre] = 0
    for rating in ratings:
        if rating == features['rating']:
            new_dict[rating] = 1
        else:
            new_dict[rating] = 0
    new_dicts.append(new_dict)

json.dump(new_dicts, open('data_for_pandas.json', 'w'))
columns = ['movie_id'] + ['critic_score', 'audience_score'] + list(genres) + most_common_people + list(ratings)
df = pd.DataFrame(new_dicts)
df = df[columns]
df.to_csv('film_features.tsv', sep='\t', index=False)
print(df.columns)

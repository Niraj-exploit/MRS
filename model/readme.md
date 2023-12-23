## Fetch movie_list.pkl and similarity.pkl from mrs-notebook.ipynby

## Raw data used link

[Raw Data](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?select=tmdb_5000_credits.csv)


```bash
pip install numpy pandas ast nltk
```

## Jupyter notebook commands

```python
import numpy as np
import pandas as pd
import ast
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

movies = pd.read_csv('tmdb_5000_movies.csv')
credit = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credit, on='title')
movies.info()

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.iloc[0].genres

movies.isnull().sum()
movies.dropna()
movies.dropna(inplace = True)
movies.isnull().sum()
movies.duplicated().sum()

movies.iloc[0].genres

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

#To test function
convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
)

movies.head()

movies['keywords'] = movies['keywords'].apply(convert)


def convert_top5(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 5:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if  i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

new_df['tags'] = new_df['tags'].apply(stem)

new_df.iloc[0].tags

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

vectors

cv.get_feature_names_out()

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
similarity[1]

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

#testing function
recommend('Avatar')

import pickle
pickle.dump(new_df.to_dict(), open('movie_dict.pkl','wb'))
pickle.dump(similarity, open('similiraty.pkl','wb'))

```



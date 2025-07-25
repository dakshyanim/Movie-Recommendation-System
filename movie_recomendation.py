
#importing dependecies
import numpy as np
import pandas as pd
from pandas import Series

#load dataset
movie_data = pd.read_csv('movie_data.csv', engine='python', on_bad_lines='skip')
print(movie_data.shape)

print(movie_data.head())

movie_data.isnull().sum()

#drop useless coloumns
movie_data.drop(['homepage', 'tagline', 'budget', 'revenue', 'id','original_language', 'original_title','production_countries', 'spoken_languages','status', 'crew' ], axis=1, inplace=True)

movie_data.shape

movie_data.isnull().sum()

movie_data.dropna(inplace=True)

movie_data.isnull().sum()

#extract the release year from release_date
movie_data['release_date'] = pd.to_datetime(movie_data['release_date'])
movie_data['release_year'] = movie_data['release_date'].dt.year

movie_data.drop(['production_companies', 'release_year', 'vote_count' ], axis=1, inplace=True)

movie_data['vote_average']=movie_data['vote_average'].astype('int')
movie_data['popularity']=movie_data['popularity'].astype('int')

movie_data.head()

#PREPROCESSING

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#done to remove comma
def collapse(text):
    return text.replace(",", " ") if isinstance(text, str) else ""

for col in ['genres', 'keywords', 'cast', 'director','overview']:
    movie_data[col] = movie_data[col].apply(collapse)

movie_data['tags'] = movie_data['genres'] + " " + movie_data['keywords'] + " " + movie_data['cast'] + " " + movie_data['director']+" "+movie_data['overview']

movie_data.head()

print(movie_data['tags'][0])

#stemming
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem_text(text):
    return " ".join([ps.stem(word) for word in text.split()]) #1

movie_data['tags'] = movie_data['tags'].apply(stem_text)

#vectorization


tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(movie_data['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

#cheching similarity
similarity = cosine_similarity(vectors)

print(similarity)



movie_data.shape

from difflib import get_close_matches

# build a lowercase list once
titles = movie_data['title'].tolist()
titles_lower = [t.lower() for t in titles]

def find_best_title(query):
    q = query.lower()
    if q in titles_lower:
        return titles_lower.index(q)
   #string matching
    matches = get_close_matches(q, titles_lower, n=1, cutoff=0.6)
    if matches:
        return titles_lower.index(matches[0])
    return None

def recommend(movie_name):
    idx = find_best_title(movie_name)
    if idx is None:
        print(f" Couldn’t find '{movie_name}' in the database.")
        return

    distances = sorted(
        list(enumerate(similarity[idx])),
        key=lambda x: x[1],
        reverse=True
    )
    print(f"\nRecommendations for '{movie_data.iloc[idx].title}':")
    for i, score in distances[1:31]:
        score = similarity[idx][i[0]]
        print(f"{movie_data.iloc[i].title}  (Similarity Score: {score:.3f})")   

 





#priting the titles of all movies
movie_data['title']
#trial1
recommend('Gandhi')

import pickle

pickle.dump(movie_data,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))

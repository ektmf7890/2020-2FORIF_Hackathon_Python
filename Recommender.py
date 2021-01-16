import pandas as pd
from ast import literal_eval

# csv 파일 불러오고 데이터 전처리하는 과정
data = pd.read_csv('tmdb_5000_movies.csv')
data = data[['id', 'genres', 'keywords', 'title']]

data['genres'] = data['genres'].apply(literal_eval)
data['keywords'] = data['keywords'].apply(literal_eval)

data['genres'] = data['genres'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))
data['keywords'] = data['keywords'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))

# CountVectorizer로 변환하는 과정
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
c_vector_genres = cv.fit_transform(data['genres'])

print(f'CountVectorizer가 찾은 장르 갯수: {len(cv.get_feature_names())}')
print(f'변환 한 Matrix 크기: {c_vector_genres.shape}')
#c_vector_genres.toarray()

#코사인 유사도 계산
from sklearn.metrics.pairwise import cosine_similarity
similartiy_matrix = cosine_similarity(c_vector_genres, c_vector_genres)

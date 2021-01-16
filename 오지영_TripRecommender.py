#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


# In[3]:


data=pd.read_csv('Tripadvisor_crawling.csv', usecols=['names', 'main_category', 'ratings', 'keywords'])
data


# In[4]:


data['keywords'][data['keywords'].isna()] = ''    # NaN값을 빈 문자열로 대체

def preprocess(string):
    
# '/', '&' 같은 문자 제거
# '장소', '및', '전문', '시설' 등과 같은 수식어 제거    
# 필요한경우 띄어쓰기 제거 (e.g. 인근 지역 -> 인근지역)

    # 공통
    
    string = string.replace(' /', '')
    string = string.replace(' &', '')
    string = string.replace(' 및', '')
    string = string.replace('아쿠아리움', '수족관') # 중복제거를 위해 아쿠아리움을 수족관으로 통일
    
    # main_category
    
    string = string.replace('보트 투어', '보트투어')
    string = string.replace('수상 스포츠', '수상스포츠')
    string = string.replace('즐길 거리', '즐길거리')
    string = string.replace('야외 활동', '야외활동')
    string = string.replace('여행자 자료', '여행자자료')

    # keywords

    string = string.replace('쇼핑몰', '쇼핑')
    string = string.replace('전문 박물관', '전문박물관 박물관')
    string = string.replace('어린이 박물관', '어린이박물관 박물관')
    string = string.replace('자연사 박물관', '자연사박물관 박물관')
    string = string.replace('군사 박물관', '군사박물관 박물관')
    string = string.replace(' 장소', '') # 종교적인 장소, 교육적인 장소
    string = string.replace('인근 지역', '인근지역')
    string = string.replace('경관이 좋은 산책로', '경관좋은산책로')
    string = string.replace('유서 깊은 산책로', '유서깊은산책로')
    string = string.replace('공공기관 건물', '공공기관건물')
    string = string.replace('고대 유적', '고대유적')
    string = string.replace('군사 기지', '군사기지')
    string = string.replace('자동차 경주장', '자동차경주장')
    string = string.replace('절경 드라이브 코스', '절경드라이브코스')
    string = string.replace('하이킹 트레일', '하이킹트레일')
    string = string.replace('야생동물 서식지', '야생동물서식지')
    string = string.replace('엔터테인먼트 센터', '엔터테인먼트센터')
    string = string.replace('도자기 스튜디오', '도자기스튜디오')
    string = string.replace('방 탈출 게임', '방탈출게임 게임')
    string = string.replace('기타 놀이', '게임') 
    string = string.replace('미니 골프', '미니골프')
    string = string.replace('물건 찾기 게임', '물건찾기게임 게임')
    string = string.replace('테마 파크', '테마파크')
    string = string.replace('대중교통 시스템', '대중교통시스템')
    
    return string

data['main_category'] = data['main_category'].apply(preprocess)
data['keywords'] = data['keywords'].apply(preprocess)





def leave_space(string):
    return ' ' + string

data['keywords_aggregated'] = data['main_category'] + data['keywords'].apply(leave_space)

data = data.drop(['main_category', 'keywords'], axis=1)

data = data.rename(columns = {'keywords_aggregated' : 'keywords'})


# 중복되는 단어 처리

def duplicates_handle(string):
    string_list = string.split(' ')
    
    tmp = []
    
    to_del = []
    
    for val in string_list:
        if string_list.count(val)!=1:
            to_del.append(val)
            tmp.append(val)

    for val in to_del:
        string = string.replace(val, '')
    
    tmp_str = ' '
    
    for val in list(set(tmp)):
        tmp_str = tmp_str + val
    
    string = string + tmp_str
    
    return string


words = ['스포츠', '게임', '박물관', '공원', '쇼핑']

for word in words:
    data['keywords'] = data['keywords'].apply(duplicates_handle)

                   
data


# In[5]:


count_vector = CountVectorizer(ngram_range=(1, 10))
c_vector_keywords = count_vector.fit_transform(data['keywords'])
c_vector_keywords.shape


# In[6]:


keyword_c_sim = cosine_similarity(c_vector_keywords, c_vector_keywords).argsort()[:, ::-1]
keyword_c_sim.shape


# In[10]:


def get_recommend_trip_list(df, place, numb, top=30):
    '''df --> 데이터프레임
    place --> 비슷한 여행지 찾고싶은 특정 여행지를 string 형식으로 입력
    numb --> 추천받고싶은 여행지 개수'''
    #비슷한 여행지를 찾고 싶은 '특정 여행지'의 인덱스 추출
    target_place_index = df[df['names'] == place].index.values
    #비슷한 코사인 유사도를 가진 여행지들의 인덱스 추출
    sim_index = keyword_c_sim[target_place_index, :top].reshape(-1)
    #자기자신 없애기
    sim_index = sim_index[sim_index != target_place_index]
    #추출한 여행지를 data frame으로 만들고 별점(ratings) 순으로 정렬한 뒤 return
    result = df.iloc[sim_index].sort_values('ratings', ascending=False)[:numb]
    print(len(result))
    return result

# 테스트
get_recommend_trip_list(df=data, place='익산미륵사지', numb=10)


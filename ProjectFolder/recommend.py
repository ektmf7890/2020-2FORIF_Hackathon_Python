
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
import random

data=pd.read_csv('Tripadvisor_crawling.csv', usecols=['names', 'main_category', 'ratings', 'keywords'])

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


count_vector = CountVectorizer(ngram_range=(1, 10))
c_vector_keywords = count_vector.fit_transform(data['keywords'])

keyword_c_sim = cosine_similarity(c_vector_keywords, c_vector_keywords).argsort()[:, ::-1]
keyword_c_sim.shape


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

    return result

num_of_sites = data.shape[0]
has_been_displayed = list(np.zeros(num_of_sites))
chosen_sites = []

def add_site_info(i, choices):
    if has_been_displayed[i] == 0:
        site_info = {
            'index': i,
            'name': data.iloc[i, :]['names'],
            'keywords': data.iloc[i, :]['keywords'],
            'ratings': data.iloc[i, :]['ratings']
        }
        choices.append(site_info)
        has_been_displayed[i] = 1


def print_site_info(site, num=None):
    if num:
        print('{:=^30}'.format(f'<{site["name"]}({num})>'))
        print(site['keywords'], end='\n\n')
    else:
        print('{:=^30}'.format(f'<{site["name"]}>'))
        print(site['keywords'])
        print(f'별점: {round(site["ratings"], 2)}', end='\n\n')


def display_choices():
    choices = []
    
    for i in random.sample(range(0, num_of_sites), 10):
        add_site_info(i, choices)
        
    while len(choices) < 10:
        add_site_info(i, choices)

    num = 1
    for choice in choices:
        print_site_info(choice, num)
        num += 1
    return choices

indices = pd.Series(data.index, index=data['names'])
def recommend():
    print('\n\n✈✈✈✈✈✈ 여행지추천해듀오 ✈✈✈✈✈✈\n 다음 여행지 중 즐겁게 다녀왔던 적이 있는 여행지가 있다면 선택해주세요😁')

    while len(chosen_sites) < 5:
        print(f'\n{5-len(chosen_sites)}개 이상을 더 선택하셔야해요😉!\n')
        choices = display_choices()
        input_list = list(map(int, input("좋았던 여행지(입력: 1 3 5, 없으면 0) : ").split()))
        
        for idx in input_list:
            if idx != 0:
                chosen_sites.append(choices[idx-1])
        
    print("\n\n🎇여행지 추천이 완료되었습니다!🎇\n")
    for site in chosen_sites:
        print(f"Because you liked {site['name']}...")
        sim_site_list = get_recommend_trip_list(df=data, place=site['name'], numb=3)
        
        for i in range(3):
            sim_site = sim_site_list.iloc[i]
            sim_dict = {
                'name' : sim_site['names'],
                'keywords': sim_site['keywords'],
                'ratings': sim_site['ratings']
            }
            print_site_info(sim_dict)
        # for sim_site in sim_site_list:
        #     print(sim_site)
        #     idx = indices[sim_site]
        #     site = {
        #         'name': data.iloc[idx, :]['names'],
        #         'keywords': data.iloc[idx, :]['keywords'],
        #         'ratings': data.iloc[idx, :]['ratings']
        #     }
        #     print_site_info(site)
        print()
            
        
recommend()

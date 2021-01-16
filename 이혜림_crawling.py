from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.webdriver.common.keys import Keys
import re
import csv
import math

#셀레니움 두둥
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.implicitly_wait(25)


#main
for i in range(6, 12):
    url = "https://www.tripadvisor.co.kr/Attractions-g294196-Activities-oa" + str(i*30) + "-South_Korea.html"
    driver.get(url)
    
    #find all 관광지 a태그, 메인 카테고리
    a_elements = driver.find_elements_by_xpath('//a[@class="_1QKQOve4"]')
    main_category = driver.find_elements_by_xpath('//span[@class="_21qUqkJx"]')

    for j in range(len(a_elements)):
        temp = []
        main_category_ = main_category[j].text
        a_elements[j].click()
        
        time.sleep(5)
        driver.switch_to.window(driver.window_handles[-1])
        
        #이름(name)
        name = driver.find_element_by_xpath('//h1[@id="HEADING"]').text
        temp.append(name)
        
        #메인 카테고리(main_category)
        temp.append(main_category_)
        
        #별점(rating)
#         ratings_ = driver.find_element_by_xpath("//span[starts-with(@class,'_3WF_jKL7')]").text
#         ratings = int(re.findall("\d+", ratings_)[0])
        
        ratingnums = driver.find_elements_by_xpath('//span[@class="_3fVK8yi6"]')
        
        review_total = 0
        ratings = 0
        
        for i in range(len(ratingnums)):
            num = int("".join(re.findall("\d+", ratingnums[i].text)))
            ratings = ratings + num
            review_total = review_total + num * review[i]
        
        rating = round(review_total/ratings, 2)
        temp.append(rating)
        
        #키워드(keyword)
        keyword_ = driver.find_elements_by_xpath('//a[@class="_1cn4vjE4"]')
        
        keyword = []
        for i in range(len(keyword_)):
            keyword.append(keyword_[i].text)
            
        temp.append(" ".join(keyword))
        print(temp)
        
        #저장
        search_list.append(temp)
        
        time.sleep(3)
        #현재 창 종료
        driver.close()
        
        # 메인 탭으로 변경
        driver.switch_to.window(driver.window_handles[0])
        

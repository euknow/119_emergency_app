import pandas as pd
import numpy as np
import datetime
import joblib
from haversine import haversine
from urllib.parse import quote
import streamlit as st
from streamlit_folium import st_folium
import folium
import branca
from geopy.geocoders import Nominatim
import ssl
from urllib.request import urlopen
import plotly.express as px
import pickle
from folium import plugins
import json
import requests
import branca
import googlemaps
import polyline
import time
import urllib.request
import lightgbm as lgb
from lightgbm import LGBMClassifier
from urllib.parse import urlencode, unquote
# import time
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys  # 엔터키를 입력할 때, 값..?
# from selenium.webdriver.common.by import By  # 
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager


#--------------------------------------------------------------------------------

# 현재 위치의 위도와 경도 그리고 입력한 병원의 위도와 경도를
# 크롬 드라이버를 이용해 구글맵에서 출발지로 도착지로 설정하여 클릭
def google_chrome(patient, hospital):

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    # 즉각적으로 크롬드라이버를 다운받아 cache 저장하여 사용하는 코드, 디폴트
    URL = 'https://www.google.co.kr/maps/dir//?hl=ko'  # 해당 페이지 주소

    driver.get(url=URL) # URL을 열기
    driver.implicitly_wait(time_to_wait=10) # max 10초까지 기다려주고 그 전에 완료되면 진행
    keyElement1 = driver.find_element(By.XPATH, '//*[@id="sb_ifc50"]/input') #구글 이미지 검색창의 XPATH, 주기적으로 바뀌니 확인할 것
    keyElement2 = driver.find_element(By.XPATH, '//*[@id="sb_ifc51"]/input')

    time.sleep(1) # 기다려주기 
    keyElement1.send_keys(patient)  # "월드컵" 이라는 단어를 검색창에 입력

    time.sleep(1) # 기다려주기 
    keyElement2.send_keys(hospital)  # "월드컵" 이라는 단어를 검색창에 입력
    keyElement2.send_keys(Keys.RETURN)  # 엔터 눌러주는 의미



#--------------------------------------------------------------------------------


def geocoding(address):
    geolocator = Nominatim(user_agent='eunho')
    location = geolocator.geocode(address)
    lati = location.latitude
    long = location.longitude
     
    return lati, long


# preprocessing : '발열', '고혈압', '저혈압' 조건에 따른 질병 전처리 함수(미션3 참고)
# 리턴 변수(중증질환,증상) : X, Y
def preprocessing(desease):
    
    desease['발열'] = new_x["체온"].map(lambda x:1 if x >= 37 else 0)
    desease['고혈압'] = new_x["수축기 혈압"].map(lambda x:1 if x >= 140 else 0)
    desease['저혈압'] = new_x["수축기 혈압"].map(lambda x:1 if x <= 90 else 0)

    Y = desease[["중증질환"]]
    X = desease[['체온', '수축기 혈압', '이완기 혈압', '호흡 곤란',
                 '간헐성 경련', '설사', '기침', '출혈', '통증', '만지면 아프다',
                 '무감각', '마비', '현기증', '졸도', '말이 어눌해졌다', '시력이 흐려짐',
                 '발열', '고혈압', '저혈압']]
                 

    return X, Y


# find_hospital : 실시간 병원 정보 API 데이터 가져오기 (미션1 참고)
# 리턴 변수(거리, 거리구분) : distance_df
def find_hospital(special_m, lati, long):
    
    context=ssl.create_default_context()
    context.set_ciphers("DEFAULT")
    
    key = "jbMloHktKBRTicjzz1JUQVaJ1j2MY%2Fyg2J764zZGrnn8i4D7q8ftgliJcERxZt8O8eyqnbv8vmTxdRmPUCNAbA%3D%3D"

    # city = 대구광역시, 인코딩 필요
    city = quote("대구광역시")

    # 미션1에서 저장한 병원정보 파일 불러오기 
    solution_df = pd.read_csv("https://raw.githubusercontent.com/euknow/Data/main/daegu_hospital_list(link).csv")

    # 응급실 실시간 가용병상 조회
    url_realtime = 'https://apis.data.go.kr/B552657/ErmctInfoInqireService/getEmrrmRltmUsefulSckbdInfoInqire'+'?serviceKey=' + key + '&STAGE1=' + city + '&pageNo=1&numOfRows=1000'
    result = urlopen(url_realtime, context=context)
    emrRealtime = pd.read_xml(result, xpath='.//item')
    solution_df = pd.merge(solution_df, emrRealtime[['hpid', 'hvec', 'hvoc']], on="hpid", how="inner")

    # 응급실 실시간 중증질환 수용 가능 여부
    url_acpt = 'https://apis.data.go.kr/B552657/ErmctInfoInqireService/getSrsillDissAceptncPosblInfoInqire' + '?serviceKey=' + key + '&STAGE1=' + city + '&pageNo=1&numOfRows=100'
    result = urlopen(url_acpt, context=context)
    emrAcpt = pd.read_xml(result, xpath='.//item')
    emrAcpt = emrAcpt.rename(columns={"dutyName":"hpid"})
    solution_df = pd.merge(solution_df,
                           emrAcpt[['hpid', 'MKioskTy1', 'MKioskTy2', 'MKioskTy3', 'MKioskTy4', 'MKioskTy5', 'MKioskTy7',
                                'MKioskTy8', 'MKioskTy9', 'MKioskTy10', 'MKioskTy11']])
                  
    # 컬럼명 변경
    column_change = {'hpid': '병원코드',
                     'dutyName': '병원명',
                     'dutyAddr': '주소',
                     'dutyTel3': '응급연락처',
                     'wgs84Lat': '위도',
                     'wgs84Lon': '경도',
                     'hperyn': '응급실수',
                     'hpopyn': '수술실수',
                     'hvec': '가용응급실수',
                     'hvoc': '가용수술실수',
                     'MKioskTy1': '뇌출혈',
                     'MKioskTy2': '뇌경색',
                     'MKioskTy3': '심근경색',
                     'MKioskTy4': '복부손상',
                     'MKioskTy5': '사지접합',
                     'MKioskTy7': '응급투석',
                     'MKioskTy8': '조산산모',
                     'MKioskTy10': '신생아',
                     'MKioskTy11': '중증화상'
                     }
    solution_df = solution_df.rename(columns=column_change)
    solution_df = solution_df.replace({"정보미제공": "N"})

    # 응급실 가용율, 포화도 추가
    
    solution_df.loc[solution_df['가용응급실수'] < 0, '가용응급실수'] = 0
    solution_df.loc[solution_df['가용수술실수'] < 0, '가용수술실수'] = 0

    solution_df['응급실가용율'] =  solution_df["가용응급실수"]/solution_df["응급실수"]
    solution_df.loc[solution_df['응급실가용율'] > 1,'응급실가용율']=1
    labels = ['불가', '혼잡', '보통', '원활']
    solution_df['응급실포화도'] = pd.cut(solution_df["응급실가용율"]*100, [-1,10,30,60,101], labels=labels)

    ### 중증 질환 수용 가능한 병원 추출
    ### 미션1 상황에 따른 병원 데이터 추출하기 참고

    if special_m == "중증 아님":
        condition1 = solution_df["응급실포화도"]!="불가"
        distance_df = solution_df[condition1]
    else:
        condition1 = (solution_df[special_m] == 'Y') & (solution_df["가용수술실수"] > 1)
        condition2 = solution_df["응급실포화도"]!="불가"
        distance_df = solution_df[condition1 & condition2]

    ### 환자 위치로부터의 거리 계산
    patient = (lati, long)
    distance = [haversine(patient, j[["위도", "경도"]], unit = 'km') for _,j in distance_df.iterrows()]
    
    
#     for idx, row in distance_df.iterrows():
#         distance.append(round(haversine((row['위도'], row['경도']), patient, unit='km'), 2))
    
    ls = ['2km이내', '5km이내', '10km이내', '10km이상']
    dis = [0,2,5,10, np.inf]
    distance_df['거리'] = distance
    distance_df['거리구분'] = pd.cut(distance_df["거리"], dis, labels=ls) 
                              
    return distance_df









# -------------------------------------------------병원 조회
# 레이아웃 구성하기 


st.set_page_config(
    page_icon="🌐",
    page_title="Map",
    layout="wide",
)

wow = {}

with st.expander(" ", expanded=True):
    st.image("https://user-images.githubusercontent.com/105876194/233120116-7aa68afc-46b6-42f2-add8-37cf3f9c4466.png")

st.markdown("## 근처 병원 조회")
st.markdown("#### 병명")
mkiosk = st.radio("질환 선택", ["중증 아님", "뇌출혈", "뇌경색", "복부손상", "심근경색", '신생아', '중증화상', '사지접합',  "응급투석", "조산산모"],horizontal=True)

symp = mkiosk
    
st.markdown("#### 현재 위치")
loc = st.text_input("현재 위치", label_visibility="collapsed")
st.markdown("#### 병원 조회")

global a
with st.form(key="tab1_first"):
    searching = st.form_submit_button(label="조회")
    if  searching :
        lati, long = geocoding(loc)
        hospital_list = find_hospital(symp, lati, long)
        
        
        
        #### 필요 병원 정보 추출 
        display_column = ['병원명', "주소", "응급연락처", "응급실수", "수술실수", "가용응급실수", "가용수술실수", '응급실포화도', '거리', '거리구분', "위도", "경도","link"]
        display_df = hospital_list[display_column]
        # display_df.reset_index(drop=True, inplace=True)
        
        #### 추출 병원 지도에 표시
        patient = [lati, long]
        chrome_patient = ",".join([str(patient[0]), str(patient[1])])

        with st.expander("인근 병원 리스트", expanded=True):
            # st.dataframe(display_df)

            ll = list(zip(list(display_df['위도']), list(display_df['경도'])))

            # 지도 중심 설정
            m = folium.Map(location=[lati, long], tiles="cartodbpositron", zoom_start=11)
            distances = []
            
            
#-----------------------------------------------------------------------------------------
            tooltip = "Patient!"
            pat_icon = folium.Icon(color="red")
            pati=f"""
                    <h4>Patient Loc!</h4><br>
                    <p>Lati : {patient[0]}</p>
                    <p>Long : {patient[1]}</p>"""

            folium.Marker(patient,
                          popup=pati,
                          icon=pat_icon,
                          tooltip=tooltip).add_to(m)    

            # 병원별 위경도에 따른 지도 시각화
            t_distances=[]
            t_times=[]
            t_locations=[]
            for i in ll:
                origin = (lati, long)
                destination = i

                # 필수 입력 파라미터
                headers = {
                    'version' : '1', 
                    'appkey' :'FHSdnK9wIb588jwExx4zy6Pl4AaDVpUt1XS84McY'
                }

                # url 입력
                url = 'https://apis.openapi.sk.com/tmap/routes'

                # queryString 입력 
                queryString = "?" + urlencode(
                    {
                     "endX": destination[1] ,     # 도착지 경도
                     "endY": destination[0] ,     # 도착지 위도
                     "startX": origin[1] ,   # 출발지 경도
                     "startY": origin[0]     # 도착지 위도
                    }
                )

                # 최종 요청 url 생성
                queryURL = url + queryString

                # API 호출
                response = requests.post(queryURL, headers=headers)

                # 딕셔너리 형태로 변환
                r_dict = json.loads(response.text) # 우리가 필요한 데이터 부문만 가져와 저장


                # 포인트, 라인, 시간, 거리 정보만 가져오기
                car_route_dict = r_dict['features']

                # 데이터프레임으로 생성
                car_route = pd.json_normalize(car_route_dict)

                # 거리, 시간 데이터 추출
                total_distance = round(car_route['properties.totalDistance'][0] / 1000 , 2)  # 경로 총 길이 (단위: m)
                total_time = round(car_route['properties.totalTime'][0] / 60, 2)             # 경로 총 소요 시간 (단위: 초)
                t_distances.append(float(total_distance))
                t_times.append(float(total_time))
                # linestring 타입만 추출
                car_route_line = car_route[car_route['geometry.type']=='LineString']
                
                geo_coord=[]
                for i, row in car_route_line.iterrows():
                    coordinates = [[c[1], c[0]] for c in row['geometry.coordinates']]
                    geo_coord.append(coordinates)
                
                t_locations.append(geo_coord)
                    
                    # plugins.AntPath(
                    # locations=[coordinates],
                    # reverse=False, # 방향 True 위에서 아래, False 아래에서 위
                    # dash_array=[20, 20],
                    # color='blue').add_to(m)

            display_df["t_times"]=t_times
            display_df["t_distances"]=t_distances
            display_df["t_locations"] = t_locations
            display_df.sort_values(["t_times", "t_distances", "응급실포화도"], ascending=[True, True, False], inplace=True)
            display_df.reset_index(drop=True, inplace=True)
            st.write(display_df)
            
            for i, row in display_df.iterrows():
                hp_loc = list(row[["위도","경도"]].values)
                dis = row["t_distances"]
                t_time = row["t_times"]
                name = row["병원명"]

                html = """<!DOCTYPE html>
                                <html>
                                    <table style="height: 126px; width: 330px;">  <tbody> <tr>
                                    <td style="background-color: #2A799C;"><div style="color: #ffffff;text-align:center;">병원명</div></td>
                                    <td style="width: 200px;background-color: #C5DCE7;">{}</td>""".format(name) + """ </tr> 
                                    <tr><td style="background-color: #2A799C;"><div style="color: #ffffff;text-align:center;">가용응급실수</div></td>
                                    <td style="width: 200px;background-color: #C5DCE7;">{}</td>""".format(row['가용응급실수']) + """</tr>
                                    <tr><td style="background-color: #2A799C;"><div style="color: #ffffff;text-align:center;">거리(km)</div></td>
                                    <td style="width: 200px;background-color: #C5DCE7;">{:.2f}</td>""".format(dis) + """ </tr>
                                    <tr><td style="background-color: #2A799C;"><div style="color: #ffffff;text-align:center;">총 소요시간(분)</div></td>
                                    <td style="width: 200px;background-color: #C5DCE7;">{}</td>""".format(t_time) + """</tr>
                                    <a href={} target=_blank>Homepage Link!</a>""".format(row['link']) + """ 
                                </tbody> </table> </html> """


                iframe = branca.element.IFrame(html=html, width=350, height=150)
                popup_text = folium.Popup(iframe, parse_html=True)
                
                
                
                if i == 0 :
                    chrome_hospital = ",".join([str(row["위도"]), str(row["경도"])])
                    
                    plugins.AntPath(locations=row["t_locations"],
                                    reverse=False, # 방향 True 위에서 아래, False 아래에서 위
                                    dash_array=[20, 20],
                                    color='red').add_to(m)
                    
                    icon_hos = plugins.BeautifyIcon(icon="star",
                                                    border_color="red",
                                                    text_color="red",
                                                    icon_shape="circle")
                    
                    folium.Marker(location=hp_loc,
                              popup=popup_text, tooltip=row['병원명'], icon=icon_hos).add_to(m)
                else:
                    plugins.AntPath(locations=row["t_locations"],
                                    reverse=False, # 방향 True 위에서 아래, False 아래에서 위
                                    dash_array=[20, 20],
                                    color='blue').add_to(m)                    
                    
                    icon_hos = plugins.BeautifyIcon(icon="star",
                                                    border_color="blue",
                                                    text_color="blue",
                                                    icon_shape="circle")
                    
                    folium.Marker(location=hp_loc,
                              popup=popup_text, tooltip=row['병원명'], icon=icon_hos).add_to(m)                    
            
            # wow["patient"] = chrome_patient
            # wow["hospital"] = chrome_hospital
            st_folium(m, width=1000)

            # best_navi = st.radio("병원을 선택하세요", ("G"))
            
            # if best_navi :
                # google_chrome(patient=chrome_patient, hospital=chrome_hospital)
                
# with st.form(key="navigator"):
#     searching = st.form_submit_button(label="가장 빠른 곳으로 안내하기")
#     if searching :
#         st.write(wow.get("patient"), wow.get("hospital"))
#         # google_chrome(patient=chrome_patient, hospital=chrome_hospital)
            
            
            
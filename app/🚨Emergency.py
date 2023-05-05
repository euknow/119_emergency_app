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

# url = "https://github.com/kimseongukk/python/raw/main/model.pickle"
# response = requests.get(url, stream=True)
# lightgbm = pickle.load(response.raw)

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

def predict_disease(new_dispatch, model):
    new_data = pd.DataFrame(new_dispatch)
    new_x = new_data[['체온', '수축기 혈압', '이완기 혈압', '호흡 곤란', '간헐성 경련', '설사', '기침', '출혈', '통증', '만지면 아프다','무감각', '마비', '현기증', '졸도', '말이 어눌해졌다', '시력이 흐려짐']]
    new_y = new_data[["중증질환"]]
    new_x['발열'] = new_x["체온"].map(lambda x:1 if x >= 37 else 0)
    new_x['고혈압'] = new_x["수축기 혈압"].map(lambda x:1 if x >= 140 else 0)
    new_x['저혈압'] = new_x["수축기 혈압"].map(lambda x:1 if x <= 90 else 0)
    
    return sym_list[np.argmax(model.predict(new_x))]

def pred_dis(df, model):
    sym_list = ['뇌경색', '뇌출혈', '복부손상', '심근경색']
    X = df[['체온', '수축기 혈압', '이완기 혈압', '호흡 곤란', '간헐성 경련', '설사', '기침', '출혈', '통증', '만지면 아프다','무감각', '마비', '현기증', '졸도', '말이 어눌해졌다', '시력이 흐려짐']]
    y = df[["중증질환"]]
    sym_list[np.argmax(model.predict(X))]    
    
    return sym_list[np.argmax(model.predict(X))]


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
    solution_df['응급실포화도'] = pd.cut(proto["응급실가용율"]*100, [-1,10,30,60,101], labels=labels)

    ### 중증 질환 수용 가능한 병원 추출
    ### 미션1 상황에 따른 병원 데이터 추출하기 참고

    if special_m == "중증 아님":
        condition1 = solution_df["응급실포화도"]!="불가"
        distance_df = solution_df[condition1]
    else:
        condition1 = (solution_df[mkiosk] == 'Y') & (solution_df["가용수술실수"] > 1)
        condition2 = solution_df["응급실포화도"]!="불가"
        distance_df = solution_df[condition1 & condition2]

    ### 환자 위치로부터의 거리 계산
    distance = [haversine(patient, j[["위도", "경도"]], unit = 'km') for _,j in solution_df.iterrows()]
    patient = (lati, long)
    
#     for idx, row in distance_df.iterrows():
#         distance.append(round(haversine((row['위도'], row['경도']), patient, unit='km'), 2))
    
    labels = ['2km이내', '5km이내', '10km이내', '10km이상']
    dis = [0,2,5,10, np.inf]
    distance_df['거리'] = distance
    distance_df['거리구분'] = pd.cut(distance_df["거리"], dis, labels=labels) 
                              
    return distance_df


#-------------------------------------구성 시작-------------------------------------------------------



## 오늘 날짜
now_date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)
now_date2 = datetime.datetime.strptime(now_date.strftime("%Y-%m-%d"), "%Y-%m-%d")

## 출동 이력의 최소 날짜, 최대 날짜
min_date = datetime.datetime.strptime("2023-01-01", "%Y-%m-%d")
max_date = datetime.datetime.strptime("2023-12-31", "%Y-%m-%d")
today_date = now_date.strftime("%Y-%m-%d")
hour_min = now_date.strftime("%H:%M")


# 레이아웃 구성하기 
st.set_page_config(
    page_icon="🚨",
    page_title="Emergency",
    layout="wide",
)

# tabs 만들기 
tab1, tab2 = st.tabs(["출동 일지", "대시보드"])

# tab1 내용물 구성하기 
with tab1:

    # 제목 넣기
    st.markdown("## 환자정보 입력 및 증상 체크")

    
    # 환자정보 널기
    st.markdown("#### 환자 정보")

    
    
## -------------------- ▼ 1-3그룹 체온/환자위치(주소) 입력 cols 구성(체온/체온 숫자 입력(fever)/환자 위치/환자위치 텍스트 입력(location)) ▼ --------------------
    
## -------------------- ▼ 1-1그룹 날짜/시간 입력 cols 구성(출동일/날짜정보(input_date)/출동시간/시간정보(input_time)) ▼ --------------------
     
    col110, col111, col112, col113 = st.columns([0.1, 0.3, 0.1, 0.3])
    with col110:
        st.info("출동일")
    with col111:
        input_date = st.date_input('출동 일자', label_visibility="collapsed")
    with col112:
        st.info("출동시간")
    with col113:
        input_time = st.time_input('출동 시간', datetime.time(now_date.hour, now_date.minute), label_visibility="collapsed")

    ## -------------------------------------------------------------------------------------


    ## -------------------- ▼ 1-2그룹 이름/성별 입력 cols 구성(이름/이름 텍스트 입력(name)/나이/나이 숫자 입력(age)/성별/성별 라디오(patient_s)) ▼ --------------------

    col120, col121, col122, col123, col124, col125 = st.columns([0.1, 0.3, 0.1, 0.1, 0.1, 0.1])
    with col120:
        st.info("이름")
    with col121:
        name = st.text_input("이름", label_visibility="collapsed")
    with col122:
        st.info("나이")
    with col123:
        age = st.number_input("나이", label_visibility="collapsed", min_value=0, max_value=120, value=26)
    with col124:
        st.info("성별")
    with col125:
        patient_s = st.radio("성별", ["남성", "여성"], label_visibility="collapsed", horizontal=True)

    ##-------------------------------------------------------------------------------------
    


    ## -------------------- ▼ 1-3그룹 체온/환자위치(주소) 입력 cols 구성(체온/체온 숫자 입력(fever)/환자 위치/환자위치 텍스트 입력(location)) ▼ --------------------
    
    col130, col131, col132, col133 = st.columns([0.1, 0.3, 0.1, 0.3])
    with col130:
        st.info("체온")
    with col131:
        fever = st.number_input("체온", min_value=30.0, max_value=50.0, label_visibility="collapsed", step=0.1,value=36.5)
    with col132:
        st.info("환자 위치")
    with col133:
        location = st.text_input("환자 위치", label_visibility="collapsed", value="대구광역시 북구 연암로 40")
    
    ##-------------------------------------------------------------------------------------

    ## ------------------ ▼ 1-4그룹 혈압 입력 cols 구성(수축기혈압/수축기 입력 슬라이더(high_blood)/이완기혈압/이완기 입력 슬라이더(low_blood)) ▼ --------------------
    ## st.slider 사용

    col140, col141, col142, col143 = st.columns([0.1, 0.3, 0.1, 0.3])
    with col140:
        st.info("수축기 혈압")
    with col141:
        high_blood = st.slider('수축기 혈압',  min_value=10, max_value=200, value=120, step=1, label_visibility="collapsed") # 140이상 고혈압, 90이하 저혈압
    with col142:
        st.info("이완기 혈압")
    with col143:
        low_blood = st.slider('수축기 혈압',  min_value=10, max_value=200, value=80, step=1, label_visibility="collapsed") # 90이상 고혈압, 60이하 저혈압

    st.markdown("#### 증상 체크하기")
    
    
    col150, col151, col152, col153, col154, col155, col156, col157 = st.columns([0.1, 0.1,0.1,0.1,0.1, 0.1,0.1,0.1])
    
    with col150:
        st.error('증상 체크')
        
    with col151:
        cough_check = st.checkbox('기침')
        convulsion_check = st.checkbox('간헐적 경련')
    
    with col152:
        paralysis_check = st.checkbox('마비')
        insensitive_check = st.checkbox('무감각')
        
    with col153:
        pain_check = st.checkbox('통증')
        touch_pain_check = st.checkbox('만지면 아픔')
    
    with col154:
        inarticulate_check = st.checkbox('말이 어눌해짐')
        swoon_check = st.checkbox('졸도')
        
    with col155:
        diarrhea_check = st.checkbox('설사')
        bleeding_check = st.checkbox('출혈')
        
    with col156:
        blurred_check = st.checkbox('시력 저하')
        breath_check = st.checkbox('호흡 곤란')
    
    with col157:
        dizziness_check = st.checkbox('현기증')
        
    patient_data = {
                    "체온": [fever],
                    "수축기 혈압": [high_blood],
                    "이완기 혈압": [low_blood],
                    "호흡 곤란": [int(breath_check)],
                    "간헐성 경련": [int(convulsion_check)],
                    "설사": [int(diarrhea_check)],
                    "기침": [int(cough_check)],
                    "출혈": [int(bleeding_check)],
                    "통증": [int(pain_check)],
                    "만지면 아프다": [int(touch_pain_check)],
                    "무감각": [int(insensitive_check)],
                    "마비": [int(paralysis_check)],
                    "현기증": [int(dizziness_check)],
                    "졸도": [int(swoon_check)],
                    "말이 어눌해졌다": [int(inarticulate_check)],
                    "시력이 흐려짐": [int(blurred_check)],
                    "중증질환": [""]}
    
    new = pd.DataFrame(patient_data)
    
    new['발열'] = new["체온"].map(lambda x:1 if x >= 37 else 0)
    new['고혈압'] = new["수축기 혈압"].map(lambda x:1 if x >= 140 else 0)
    new['저혈압'] = new["수축기 혈압"].map(lambda x:1 if x <= 90 else 0)    

    st.dataframe(new)
    
    st.markdown("## 증상 예측")
        

    
    col211, col212, col213 = st.columns([0.1, 0.35, 0.35])
    
    with col211:
        st.error("증상 예측 여부")
    
    with col212:
        special_yn = st.selectbox("중증",
                      ('중증 질환 예측 안함', "중증 질환 예측"), label_visibility="collapsed")
    

                
## -------------------- ▼ 1-7그룹 중증 질환 선택 또는 예측 결과 표시 cols 구성 ▼ --------------------        
    url = 'https://github.com/euknow/Data/blob/main/Lightmodel?raw=true'   
    response = requests.get(url, stream=True)
    lightgbm = pickle.load(response.raw)
    
    ## 모델 불러오기
    # lightgbm = pickle.load(open("C:/Users/User/박은호/KT AIVLE SCHOOL/미니프로젝트5/2일차/LightGBM", 'rb'))
    
    col61, col62 = st.columns([0.5, 0.5])
    with col61:
        button1 = st.button("complete")
        if button1 : 
            if  special_yn == "중증 질환 예측":
                X = new[['체온', '수축기 혈압', '이완기 혈압', '호흡 곤란', '간헐성 경련', '설사', '기침', '출혈', '통증', '만지면 아프다','무감각', '마비', '현기증', '졸도', '말이 어눌해졌다', '시력이 흐려짐',
                         "발열", "고혈압", "저혈압"]]
                y = new[["중증질환"]]
                sym_list = ['뇌경색', '뇌출혈', '복부손상', '심근경색']
                special_m = sym_list[np.argmax(lightgbm.predict(X))]
                st.markdown(f"### 예측된 중증 질환은 {special_m}입니다")
                st.write("중증 질환 예측은 뇌출혈, 뇌경색, 심근경색, 응급내시경 4가지만 분류됩니다.")
                st.write("다음 페이지로 가서 적합한 병원을 조회하세요.")
                st.write("이외의 중증 질환으로 판단될 경우, 직접 선택하세요")

            else:
                st.markdown(f"### 밑에서 질환을 선택하세요!")
                
    st.markdown("## 데이터 추가")

    special_m = st.radio("중증 질환 선택", ['뇌출혈', '신생아', '중증화상', "뇌경색", "심근경색", "복부손상", "사지접합",  "응급투석", "조산산모"],horizontal=True, label_visibility="collapsed")
                
        ##  -------------------- ▼ 1-9그룹 완료시간 저장 폼 지정 ▼  --------------------
    with st.form(key='tab1_second'):

        ## 완료시간 시간표시 cols 구성
        col191, col192 = st.columns(2)
        
        with col191:
            st.success("완료 시간")
        with col192:
            end_time = st.time_input('완료 시간', datetime.time(now_date.hour, now_date.minute), label_visibility="collapsed")

        ## 완료시간 저장 버튼
        if st.form_submit_button(label='저장하기'):
            dispatch_data = pd.read_csv("https://raw.githubusercontent.com/euknow/Data/main/119_emergency_dispatch.csv", encoding="cp949" )
            id_num = list(dispatch_data['ID'].str[1:].astype(int))
            max_num = np.max(id_num)
            max_id = 'P' + str(max_num)
            elapsed = (end_time.hour - input_time.hour)*60 + (end_time.minute - input_time.minute)

            check_condition1 = (dispatch_data.loc[dispatch_data['ID'] ==max_id, '출동일시'].values[0]  == str(input_date))
            check_condition2 = (dispatch_data.loc[dispatch_data['ID']==max_id, '이름'].values[0] == name)

            ## 마지막 저장 내용과 동일한 경우, 내용을 update 시킴
            
            if check_condition1 and check_condition2:
                dispatch_data.loc[dispatch_data['ID'] == max_id, '나이'] = age
                dispatch_data.loc[dispatch_data['ID'] == max_id, '성별'] = patient_s
                dispatch_data.loc[dispatch_data['ID'] == max_id, '체온'] = fever
                dispatch_data.loc[dispatch_data['ID'] == max_id, '수축기 혈압'] = high_blood
                dispatch_data.loc[dispatch_data['ID'] == max_id, '이완기 혈압'] = low_blood
                dispatch_data.loc[dispatch_data['ID'] == max_id, '호흡 곤란'] = int(breath_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '간헐성 경련'] = int(convulsion_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '설사'] = int(diarrhea_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '기침'] = int(cough_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '출혈'] = int(bleeding_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '통증'] = int(pain_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '만지면 아프다'] = int(touch_pain_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '무감각'] = int(insensitive_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '마비'] = int(paralysis_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '현기증'] = int(dizziness_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '졸도'] = int(swoon_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '말이 어눌해졌다'] = int(inarticulate_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '시력이 흐려짐'] = int(blurred_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '중증질환'] = special_m
                dispatch_data.loc[dispatch_data['ID'] == max_id, '이송 시간'] = int(elapsed)

            else: # 새로운 출동 이력 추가하기
                new_id = 'P' + str(max_num+1)
                new_data = {
                    "ID" : [new_id],
                    "출동일시" : [str(input_date)],
                    "이름" : [name],
                    "성별" : [patient_s],
                    "나이" : [age],
                    "체온": [fever],
                    "수축기 혈압": [high_blood],
                    "이완기 혈압": [low_blood],
                    "호흡 곤란": [int(breath_check)],
                    "간헐성 경련": [int(convulsion_check)],
                    "설사": [int(diarrhea_check)],
                    "기침": [int(cough_check)],
                    "출혈": [int(bleeding_check)],
                    "통증": [int(pain_check)],
                    "만지면 아프다": [int(touch_pain_check)],
                    "무감각": [int(insensitive_check)],
                    "마비": [int(paralysis_check)],
                    "현기증": [int(dizziness_check)],
                    "졸도": [int(swoon_check)],
                    "말이 어눌해졌다": [int(inarticulate_check)],
                    "시력이 흐려짐": [int(blurred_check)],
                    "중증질환": [special_m],
                    "이송 시간" : [int(elapsed)]
                }

                new_df= pd.DataFrame(new_data)
                dispatch_data = pd.concat([dispatch_data, new_df], axis=0, ignore_index=True)

            dispatch_data.to_csv('./119_emergency_dispatch.csv', encoding="cp949", index=False)
            st.markdown("##### 데이터가 추가되었습니다.")

    # -------------------- 완료시간 저장하기 END-------------------- 
    
    
    
    
    
    
    
    
    # -------------------- ▼ 필요 변수 생성 코딩 Start ▼ --------------------

data = pd.read_csv("https://raw.githubusercontent.com/euknow/Data/main/119_emergency_dispatch.csv", encoding="cp949" )
data = data.sort_values('출동일시', ascending=True)

## 오늘 날짜
now_date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)
now_date2 = datetime.datetime.strptime(now_date.strftime("%Y-%m-%d"), "%Y-%m-%d")

## 2023년 최소 날짜(2023-01-01), 최대 날짜(2023-12-31)
first_date = pd.to_datetime('2023-01-01')
last_date = pd.to_datetime('2023-12-31')

## 출동 이력의 최소 날짜, 최대 날짜
min_date = datetime.datetime.strptime(data['출동일시'].min(), "%Y-%m-%d")
max_date = datetime.datetime.strptime(data['출동일시'].max(), "%Y-%m-%d")


# -------------------- ▲ 필요 변수 생성 코딩 End ▲ --------------------


# -------------------- ▼ Streamlit 웹 화면 구성 START ▼ --------------------

with tab2:
    st.markdown('## 119 대시보드')

    
# tab2 내용 구성하기
 
    
    ## -------------------- ▼ 2-0그룹 금일 출동 이력 출력 ▼ --------------------
    today_date = now_date.strftime("%Y-%m-%d")
    data['datetime'] = pd.to_datetime(data['출동일시'])
    # today_count = data[data['datetime']==today_date].shape[0]
    
    
    st.info('금일 출동 내역')
    col210, col211, col212 = st.columns([0.3,0.2,0.1])
    with col210:
        slider_date = st.slider('날짜', min_value=min_date, max_value=max_date,
                               value = (min_date, now_date2))
        
    with col211:
        slider_week = st.slider('주간', min_value=min_date, max_value=max_date,
                               step=datetime.timedelta(weeks=1), value=(min_date, now_date2))
        
    with col212:
        slider_month = st.slider('월간', min_value=min_date, max_value=max_date,
                               step=datetime.timedelta(weeks=1), value=(min_date, now_date2), format='YYYY-MM')
        
    d_count = data[(slider_date[0] <= data['datetime']) & (data['datetime'] <= slider_date[1])].shape[0]
        
    if d_count > 0:
        st.dataframe(data[(slider_date[0] <= data['datetime']) & (data['datetime'] <= slider_date[1])])
    else:
        st.markdown("금일 출동내역이 없습니다.")

    
    
    
    ## -------------------------------------------------------------------

    ## -------------------- ▼ 2-1그룹 통계 조회 기간 선택하기 ▼ --------------------

    ## 선택된 일자의 data 추출
    day_list_df = data[(data['datetime'] >= slider_date[0]) & (data['datetime']<=slider_date[1])]
    

    ## 선택된 주간의 data 추출
    data['주별'] = data['datetime'].dt.strftime("%W").astype(int)
    data['주별'] = data['주별'] + 1
    
    min_week = int(slider_week[0].strftime("%W"))
    max_week = int(slider_week[1].strftime("%W"))
    week_list_df = data[(data['주별'] >= min_week) & (data['주별'] <= max_week)]
        

    ## 선택된 월의 data 추출
    
    data['월별'] = data['datetime'].dt.month.astype(int)
    min_month = slider_month[0].month
    max_month = slider_month[1].month
    
    month_list_df = data[(data['월별'] >= min_month) & (data['월별'] <= max_month)]


    ## -------------------------------------------------------------------------------------------

    ## -------------------- ▼ 2-2그룹 일간/주간/월간 총 출동 건수 통계 그래프 ▼ --------------------

    
    select_bins = st.radio("주기", ('일자별', '주별', '월별'), horizontal=True)
    st.error("출동 건수")
    
    if select_bins == '일자별':
        group_day = day_list_df.groupby(by='datetime', as_index=False)['ID'].count()
        group_day = group_day.rename(columns={'ID' : '출동건수'})
        st.bar_chart(data=group_day, x='datetime', y='출동건수',use_container_width=True)
    
    elif select_bins=='주별':

        group_week = week_list_df.groupby(by='주별', as_index=False)['ID'].count()
        group_week = group_week.rename(columns={'ID' : '출동건수'})
        group_week = group_week.sort_values('출동건수', ascending=True)
        st.bar_chart(data=group_week, x='주별', y='출동건수', use_container_width=True) 

    else:

        group_month = month_list_df.groupby(by='월별', as_index=False)['ID'].count()
        group_month = group_month.rename(columns={'ID' : '출동건수'})
        group_month = group_month.sort_values('출동건수', ascending=True)
        st.bar_chart(data=group_month, x='월별', y='출동건수', use_container_width=True)

    ## -------------------------------------------------------------------------------------------

    ## -------------------- ▼ 2-3그룹 일간/주간/월간 평균 이송시간 통계 그래프 ▼ --------------------
    


    st.success("이송시간 통계")

    col230, col231, col232 = st.columns([0.3,0.3,0.3])
    with col230:
        
        group_day_time = data.groupby(by='출동일시', as_index=False)['이송 시간'].mean()
        group_day_time = group_day_time.rename(columns={"이송 시간": '이송시간'})
        st.line_chart(data=group_day_time, x='출동일시', y='이송시간', use_container_width=True) 
        
    with col231:

        group_week_time = data.groupby(by='나이', as_index=False)['이송 시간'].mean()
        group_week_time = group_week_time.rename(columns={"이송 시간": '이송시간'})
        st.line_chart(data=group_week_time, x='나이', y='이송시간', use_container_width=True)
        

    with col232:

        group_month_time = data.groupby(by='중증질환', as_index=False)['이송 시간'].mean()
        group_month_time = group_month_time.rename(columns={"이송 시간": '이송시간'})
        st.line_chart(data=group_month_time, x='중증질환', y='이송시간', use_container_width=True)
        


        
        
        # -------------------------------------------------------------------------------------------

# ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ [도전 미션] ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ 

    ## -------------------- ▼ 2-4그룹 일간/주간/월간 중증 질환별 비율 그래프 ▼ --------------------
    
    st.warning("중증 질환별 통계")
    
    col240, col241, col242 = st.columns(3)
    
    with col240: # 일간 통계

        group_day_disease = data.groupby(by='중증질환', as_index=False)['datetime'].count()
        group_day_disease = group_day_disease.rename(columns={"datetime": '합계'})

        fig = px.pie(data_frame=group_day_disease, values='합계', names='중증질환')
        fig.update_traces(textposition='inside',textinfo='label+percent+value', hole=.3)
        fig.update_layout(title='일간 통계')
        st.plotly_chart(fig, use_container_width=True)

    with col241: # 주간 통계

        group_week_disease = data.groupby(by='중증질환', as_index=False)['주별'].count()
        group_week_disease = group_week_disease.rename(columns={"주별": '합계'})

        fig = px.pie(data_frame=group_week_disease, values='합계', names='중증질환')
        fig.update_traces(textposition='inside',textinfo='label+percent+value', hole=.3)
        fig.update_layout(title='주간 통계')
        st.plotly_chart(fig, use_container_width=True)

    with col242: # 월간 통계
        group_month_disease = data.groupby(by='중증질환', as_index=False)['월별'].count()
        group_month_disease = group_month_disease.rename(columns={"월별": '합계'})

        fig = px.pie(data_frame=group_month_disease, values='합계', names='중증질환')
        fig.update_traces(textposition='inside',textinfo='label+percent+value', hole=.3)
        fig.update_layout(title='월간 통계')
        st.plotly_chart(fig, use_container_width=True)
     

    
    

    ## -------------------------------------------------------------------------------------------

    ## -------------------- ▼ 2-5그룹 그외 필요하다고 생각되는 정보 추가 ▼ --------------------

#Here is the link to my github repsitory
#https://github.com/TioHK/final_project_dsci510

import requests
import json
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import statsmodels.api as sm
from geopy import distance
import plotly.graph_objects as go
from patsy import dmatrices
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',None)


#data query process-census data
HH_list_2020= 'SERIALNO,PUMA,ACCESSINET,BROADBND,COMPOTHX,DIALUP,HISPEED,LAPTOP,OTHSVCEX,FS,SATELLITE,SMARTPHONE,TABLET,VALP,FINCP,HHL,HINCP,HUPAOC,WGTP'
PP_list_2020='PUMA,SERIALNO,PWGTP,AGEP,HINS3,HINS4,SCHL,SEX,POVPIP,RACWHT,WRK,PAP,SSIP'
#HH_list_2019='SERIALNO,PUMA,ACCESS,BROADBND,COMPOTHX,DIALUP,HISPEED,LAPTOP,OTHSVCEX,FS,SATELLITE,SMARTPHONE,TABLET,VALP,FINCP,HHL,HINCP,HUPAOC,WGTP'
#PP_list_2019='PUMA,SERIALNO,PWGTP,AGEP,HINS3,HINS4,SCHL,SEX,POVPIP,RACWHT,WRK,PAP,SSIP'
#HH_list_2018='SERIALNO,PUMA,ACCESS,BROADBND,COMPOTHX,DIALUP,HISPEED,LAPTOP,OTHSVCEX,FS,SATELLITE,SMARTPHONE,TABLET,VALP,FINCP,HHL,HINCP,HUPAOC,WGTP'
#PP_list_2018='PUMA,SERIALNO,PWGTP,AGEP,HINS3,HINS4,SCHL,SEX,POVPIP,RACWHT,WRK,PAP,SSIP'
#HH_list_2017='SERIALNO,PUMA,ACCESS,BROADBND,COMPOTHX,DIALUP,HISPEED,LAPTOP,OTHSVCEX,FS,SATELLITE,SMARTPHONE,TABLET,VALP,FINCP,HHL,HINCP,HUPAOC,WGTP'
#PP_list_2017='PUMA,SERIALNO,PWGTP,AGEP,HINS3,HINS4,SCHL,SEX,POVPIP,RACWHT,WRK,PAP,SSIP'

def get_data(HH_list,PP_list,year):
    api_key = '1d8047e886b89aac8dd078472849d0ff52b5f668'
    state = '06'
    base_url = f'https://api.census.gov/data/{year}/acs/acs5/pums'
    HH_url = f'{base_url}?get={HH_list}&for=state:{state}&key={api_key}'
    PP_url = f'{base_url}?get={PP_list}&for=state:{state}&key={api_key}'
    HH_response=requests.get(HH_url)
    PP_response=requests.get(PP_url)
    HH_data=HH_response.json()
    PP_data=PP_response.json()
    HH_df = pd.DataFrame(HH_data[1:], columns = HH_data[0])
    PP_df = pd.DataFrame(PP_data[1:], columns = PP_data[0])
    HH_df.to_csv(f'data/CaliforniaHH_{year}.csv',index=False)
    PP_df.to_csv(f'data/CaliforniaPP_{year}.csv', index=False)
    print ('finished!')

get_data(HH_list_2020,PP_list_2020,2020)
#get_data(HH_list_2019,PP_list_2019,2019)
#get_data(HH_list_2018,PP_list_2019,2018)
#get_data(HH_list_2017,PP_list_2019,2017)


#Read the address out from the original csv file and prepared for the address used for following api query
excel_f=pd.read_excel ('data/FO-Address-Open-Close-Times.xlsx',header=7,index_col=0)
excel_f.to_csv('data/FO-Address-Open-Close-Times.csv')
location_info=pd.read_csv('data/FO-Address-Open-Close-Times.csv')
location_info.drop(location_info.columns[10:], axis=1, inplace=True)
location_info.columns=['Office Code','Office Name','Address Line 1','Address Line 2','Address Line 3','City','State','Zip','Phone','Fax']
location_info = location_info[location_info.State == 'CA']

#correction of initial errors in the provided start here
location_info.at[995,'Address Line 3']=location_info.at[995,'Address Line 2']
location_info.at[1028,'Address Line 3']=location_info.at[1028,'Address Line 2']
location_info.at[1041,'Address Line 3']=location_info.at[1041,'Address Line 2']
location_info.at[1047,'Address Line 3']=location_info.at[1047,'Address Line 2']
location_info.at[1051,'Address Line 3']=location_info.at[1051,'Address Line 2']
location_info.at[1069,'Address Line 3']=location_info.at[1069,'Address Line 2']
location_info.at[1071,'Address Line 3']=location_info.at[1071,'Address Line 2']
location_info.at[1121,'Address Line 3']=location_info.at[1121,'Address Line 2']
location_info.at[1131,'Address Line 3']=location_info.at[1131,'Address Line 2']
location_info.drop(['Address Line 1','Address Line 2'], axis=1,inplace=True)
#correction of initial errors ends here

location_info['Zip'] = location_info['Zip'].apply(str)
location_info["APIaddress"] = location_info[['Address Line 3','City','State','Zip']].apply(" ".join, axis=1)
location_info.to_csv('data/CASSO.csv')


#use the address in the csv file and make the final csv file
list_l = []
list_ln = []
api_key1='AIzaSyBR5ySmJaAZSAIwd6LUejMYhE_qWAnmR7c'
for index, row in location_info.iterrows():
    address=row['APIaddress']
    geoapi=f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key1}'
    geo_response=requests.get(geoapi)
    data=json.loads(geo_response.text)
    try:
        lat_num=data['results'][0]['geometry']['location']['lat']
        lng_num=data['results'][0]['geometry']['location']['lng']
        list_l.append(lat_num)
        list_ln.append(lng_num)
    except:
        list_l.append('address')
lat=list_l
lag=list_ln
location_info['lat']=lat
location_info['lag']=lag
location_info.reset_index(drop=True,inplace=True)
location_info.to_csv('data/foinfo.csv')


#Trim the zipcode centroid file
ca_centroid=pd.read_csv('data/ZIP_Code_Population_Weighted_Centroids.csv')
ca_centroid = ca_centroid[ca_centroid.USPS_ZIP_PREF_STATE_1221 == 'CA']
ca_centroid.reset_index(drop=True,inplace=True)
ca_centroid.to_csv('data/zipcenter.csv')


#cleaning for 2020 household data
HH_frame_2020=pd.read_csv('data/CaliforniaHH_2020.csv',low_memory=False)
HH_frame_2020.drop(HH_frame_2020[HH_frame_2020['WGTP'] == 0].index, inplace = True)
HH_frame_2020.replace({'ACCESSINET': 3, 'BROADBND':2,'COMPOTHX':2,'DIALUP':2, 'HISPEED':2,'LAPTOP':2,'OTHSVCEX':2,'SATELLITE':2,'SMARTPHONE':2,'TABLET':2,'FS':2}, 0, inplace = True)
HH_frame_2020.replace({'ACCESSINET': 2},1,inplace=True)
HH_frame_2020['HHL'].replace({2: 0,3:0,4:0,5:0},inplace=True)
HH_frame_2020['HUPAOC'].replace({2:1,3:1,4:0},inplace=True)
conditions = [
    HH_frame_2020['ACCESSINET'].eq(1), HH_frame_2020['BROADBND'].eq(1),
    HH_frame_2020['DIALUP'].eq(1), HH_frame_2020['OTHSVCEX'].eq(1),
    HH_frame_2020['SATELLITE'].eq(1)
]
choices = [1,1,1,1,1]
HH_frame_2020['Have_internet'] = np.select(conditions, choices, default=0)

conditions = [
    HH_frame_2020['SMARTPHONE'].eq(1), HH_frame_2020['TABLET'].eq(1),
    HH_frame_2020['COMPOTHX'].eq(1)
]
choices = [1,1,1]
HH_frame_2020['Have_3C'] = np.select(conditions, choices, default=0)
HH_frame_2020['VALP'] = HH_frame_2020['VALP'].mask(HH_frame_2020['VALP'] <0, 0)
HH_frame_2020['HINCP'] = HH_frame_2020['HINCP'].mask(HH_frame_2020['HINCP'] <0, 0)
HH_frame_2020['FINCP'] = HH_frame_2020['FINCP'].mask(HH_frame_2020['FINCP'] <0, 0)

#cleaning for 2020 individual data: cleaning out the household which are receiving federal assistance and merge it to the household level data file
PP_frame_2020=pd.read_csv('data/CaliforniaPP_2020.csv',low_memory=False)
PP_frame_2020.replace({'HINS4': 2},0,inplace=True)
PP_frame_2020["POVPIP"] = np.where(PP_frame_2020["POVPIP"] <= 135, 1, 0)
PP_frame_2020.replace({'PAP': -1,'SSIP': -1},0,inplace=True)
conditions = [
    PP_frame_2020['HINS4'].eq(1), PP_frame_2020['POVPIP'].eq(1),
    PP_frame_2020['PAP'].eq(1),PP_frame_2020['SSIP'].eq(1)
]
choices = [1,1,1,1]
PP_frame_2020['Eligible'] = np.select(conditions, choices, default=0)
update_PP_2020=PP_frame_2020.drop(['HINS3', 'state','PWGTP','SCHL','SEX','WRK','AGEP','RACWHT','HINS4','POVPIP','PAP','SSIP','PUMA'] ,axis=1)
update_PP_2020=update_PP_2020.groupby('SERIALNO').sum()
FUll_household_2020=pd.merge(HH_frame_2020,update_PP_2020,on='SERIALNO',how='inner')

#cleaning individual level data: non-eligibility related factors
PP_2020_new=PP_frame_2020.drop(['SERIALNO','HINS3','HINS4','POVPIP','PAP','SSIP','state','Eligible'] ,axis=1)
PP_2020_new.replace({'SEX': 2},0,inplace=True)
PP_2020_new.replace({'WRK': 2},0,inplace=True)
PP_2020_new["SCHL"] = np.where(PP_2020_new["SCHL"] <= 20, 0, 1)

#collapse and merge the data on puma level
variables=['AGEP','SCHL','SEX','RACWHT','WRK']
collpased_PP_2020=PP_2020_new.groupby('PUMA').apply(lambda PP_2020_new: pd.Series([sum(PP_2020_new[v] * PP_2020_new.PWGTP) / sum(PP_2020_new.PWGTP) for v in variables]))
collpased_PP_2020.columns=[e +'_PUMA' for e in variables]
collpased_PP_2020.reset_index(inplace=True)

variables=[
    'ACCESSINET','BROADBND','COMPOTHX','DIALUP',
    'HISPEED','LAPTOP','OTHSVCEX','SATELLITE',
    'SMARTPHONE','TABLET','VALP','FINCP','HHL',
    'HINCP','HUPAOC','Have_internet','Have_3C',
    'Eligible'
]
collpased_HH_2020 = FUll_household_2020.groupby('PUMA').apply(lambda FUll_household_2020: pd.Series([sum(FUll_household_2020[v] * FUll_household_2020.WGTP) / sum(FUll_household_2020.WGTP) for v in variables]))
collpased_HH_2020.columns = [e+'_PUMA' for e in variables]
collpased_HH_2020.reset_index(inplace=True)

final_data_set_1=pd.merge(collpased_HH_2020,collpased_PP_2020,on='PUMA',how='inner')
final_data_set_1.to_csv('data/final_data_set_1.csv')

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


#mapping preparation
map_puma_infromation=pd.read_csv('data/final_data_set_1.csv')
map_puma_infromation['PUMA']=map_puma_infromation['PUMA'].apply(lambda x: f'{x:05d}')
map_puma_infromation['PUMA1']=map_puma_infromation['PUMA']
map_puma_infromation['PUMA1']=map_puma_infromation['PUMA1'].astype(int)
map_puma_infromation.set_index('PUMA1',inplace=True)
print(map_puma_infromation.head())
puma_gpd=gpd.read_file('data/tl_2020_06_puma10/tl_2020_06_puma10.shp')
#puma_gpd['PUMACE10'] = puma_gpd['PUMACE10'].astype(int)
puma_gpd.rename(columns={'PUMACE10':'PUMA'}, inplace=True)
puma_gpd_merged=puma_gpd.merge(map_puma_infromation,on='PUMA',how='inner')
puma_gpd_merged.to_file('data/for_puma_map.geojson', driver='GeoJSON')
with open ('data/for_puma_map.geojson','r') as infile:
    pumajson=json.load(infile)


#Distance Cauculation and the second research question anaysis 
centroid=pd.read_csv('data/zipcenter.csv')
office=pd.read_csv('data/foinfo.csv')
centroid_dict=centroid.to_dict('index')
office_dict=office.to_dict('index')
#print(office_dict)
#print(centroid_dict)
distance_dict={}
for index in centroid_dict:
    tuple_centroid=(centroid_dict[index]['LAT'],centroid_dict[index]['LGT'] )
    distance_list = []
    for indexo in office_dict:
        tuple_office=(office_dict[indexo]['lat'],office_dict[indexo]['lag'])
        m=distance.distance(tuple_centroid, tuple_office).km
        distance_list.append(m)
    closest_distance=min(distance_list)
    distance_dict[centroid_dict[index]['STD_ZIP5']]=closest_distance
distance = pd.DataFrame.from_dict(distance_dict.items())
distance.columns = ['ZIPCODE', 'Distance']

participants=pd.read_csv('data/Lifelineparticipants.csv')
distance_participants=distance.merge(participants,on='ZIPCODE',how='inner')
distance_participants.to_csv('data/distance_participants')
new_distance_participants=pd.read_csv('data/distance_participants')
new_distance_participants.drop(['2015', '2016','2017','2018','2019','2020','2021','Differences','Unnamed: 0'], axis=1,inplace=True)
new_distance_participants.dropna(how='any',inplace=True)
#print(new_distance_participants.isnull().sum())
new_distance_participants['participants']=new_distance_participants['2022']
y, X = dmatrices('participants ~ Distance', data=new_distance_participants, return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())

plt.scatter(new_distance_participants.Distance, new_distance_participants.participants)
plt.title('Distance to SSO vs. Lifeline Participants')
plt.xlabel('Distance to SSO')
plt.ylabel('Lifeline Participants')
plt.savefig('data/scatterplot')
sns.regplot(x=new_distance_participants['Distance'], y=new_distance_participants['participants'],scatter_kws={"color": "black"}, line_kws={"color": "red"})
plt.savefig('data/fittedline')
plt.close()

# Data Analysis
#First Research Question Analysis

def histogram (variable):
    plt.hist(map_puma_infromation[variable],color="skyblue",range=(0,1),bins=20)
    plt.xlabel(f'{variable}')
    plt.ylabel('Count')
    plt.title(f'Histogram of {variable}')
    plt.savefig(f'data/{variable}')

histogram('Have_internet_PUMA')
histogram('Have_3C_PUMA')
histogram('HISPEED_PUMA')

def plot(column):
    puma_gpd_merged.plot(column=column,legend=True,cmap='OrRd')
    plt.savefig(f'data/{column}1')

plot('Have_internet_PUMA')
plot('Have_3C_PUMA')
plot('HISPEED_PUMA')

def regression_output (list1,y_str):
    x = map_puma_infromation[list1]
    y = map_puma_infromation[y_str]
    x = sm.add_constant(x)
    templete = sm.OLS(y, x).fit()
    pre_dic = templete.predict(x)
    OLS_output = templete.summary()
    return OLS_output

def get_VIF (list1,y_str):
    variables = "+".join(list1)
    y, X = dmatrices(y_str +'~' + variables, map_puma_infromation, return_type='dataframe')
    VIF= pd.DataFrame()
    VIF["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    VIF["Variables"] = X.columns
    return VIF

list=['VALP_PUMA','HHL_PUMA','HINCP_PUMA','Eligible_PUMA','AGEP_PUMA','SCHL_PUMA','SEX_PUMA','WRK_PUMA']
y='Have_internet_PUMA'
y1='Have_3C_PUMA'
y2='HISPEED_PUMA'

print(regression_output (list,y))
print(regression_output (list,y1))
print(regression_output (list,y2))

print(get_VIF (list,y))

list_new=['HHL_PUMA','AGEP_PUMA','SCHL_PUMA','SEX_PUMA','WRK_PUMA']

print(regression_output (list_new,y))
print(regression_output (list_new,y1))
print(regression_output (list_new,y2))

print(get_VIF (list_new,y))

#Interactive Map
app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H2('Select a internet/3-c products usage rate'),
            html.P("Select a type:"),
            dcc.Dropdown(
                id='selection',
                options=[
                    {"label": "Rate of having high speed internet", "value": 'HISPEED_PUMA'},
                    {"label": "Rate of having general internet access", "value": 'Have_internet_PUMA'},
                    {"label": "Rate of having 3-C products", "value": 'Have_3C_PUMA'}, ],
                multi=False,
                value='Have_internet_PUMA',
            )],
        style={'width': '49%', 'display': 'inline-block'},
        ),

        html.Div([
            html.H2('Related background information in each puma'),
            html.P("Please click on a PUMA in the map to get the information"),

        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='map',
            clickData={'points': [{'location': '00101'}]}
        )
    ], style={'width': '49%','display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='graph_rate'),
        dcc.Graph(id='graph_money'),
    ], style={'display': 'inline-block', 'width': '49%'}),
])


@app.callback(
    Output("map", 'figure'),
    Input("selection", "value"))
def output_graph(selection):
    map1 = px.choropleth_mapbox(map_puma_infromation, geojson=pumajson, locations='PUMA', featureidkey='properties.PUMA',
                               color=selection,
                               color_continuous_scale="viridis",
                               range_color=(0, 1),
                               mapbox_style="carto-positron",
                               zoom=4.5, center={"lat": 34.052235, "lon": -118.243683},
                               )
    map1.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},height=900)
    return map1

@app.callback(
    Output('graph_rate', 'figure'),
    Input("map", 'clickData'))
def output_graph(clickData):
    m = int(clickData['points'][0]['location'])
    Englishspeake=map_puma_infromation.at[m,'HHL_PUMA']
    Havingchildren= map_puma_infromation.at[m, 'HUPAOC_PUMA']
    Eligible = map_puma_infromation.at[m, 'Eligible_PUMA']
    College = map_puma_infromation.at[m, 'SCHL_PUMA']
    male=map_puma_infromation.at[m, 'SEX_PUMA']
    white = map_puma_infromation.at[m, 'RACWHT_PUMA']
    employment = map_puma_infromation.at[m, 'WRK_PUMA']
    fig = go.Figure(go.Bar(
        x=[Englishspeake,Havingchildren,Eligible,College,male,white,employment],
        y=['English speaker','Having children','Eligible for assistance','College Diploma','Male','White','Employed'],
        orientation='h'))
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},xaxis_title="Percent of households of corresponding characteristics (in decimal)")
    return fig

@app.callback(
    Output('graph_money', 'figure'),
    Input("map", 'clickData'))
def output_graph(clickData):
    m = int(clickData['points'][0]['location'])
    HouseholdIncome=map_puma_infromation.at[m,'HINCP_PUMA']
    propertyValue= map_puma_infromation.at[m, 'VALP_PUMA']
    fig = go.Figure(go.Bar(
        x=[HouseholdIncome,propertyValue],
        y=['Average household Income','Average property value'],
        orientation='h'))
    fig.update_layout(margin={"r": 10, "t": 10, "l": 10, "b": 10},xaxis_title='in Dollar ($)')
    return fig

app.run_server(debug=True,port=2223)

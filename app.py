#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.svm import SVR
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import math
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,  explained_variance_score, mean_squared_log_error,median_absolute_error,r2_score
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from flask import Flask
import os
import dash_daq as daq
import base64
import io
#from flask import send_file

spinner_type='graph'


number_genetive = {1:'yhden',2:'kahden',3:'kolmen'}
number_to = {1:'yhdelle',2:'kahdelle',3:'kolmelle'}

url = 'https://pxnet2.stat.fi:443/PXWeb/api/v1/fi/StatFin/vrm/vaerak/statfin_vaerak_pxt_11re.px'
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36','Content-Type':'application/json'}

variables = pd.DataFrame(requests.get(url).json()['variables']).set_index('code')

vuodet = [int(c) for c in variables.loc['Vuosi'].valueTexts]

#iät = [int(c.replace(' ','').replace('-','').replace('+','')) for c in variables.loc['Ikä'].valueTexts if len(c.replace(' ','').replace('-','').replace('+',''))<=3]

areas = requests.get(url).json()['variables'][0]
cities = pd.DataFrame(areas['values'],columns = ['aluekoodi'])
cities['alue'] = areas['valueTexts']
cities.alue = cities.alue.str.capitalize()

cities = cities.set_index('alue')

vuosi_options = [{'label':s, 'value': s} for s in sorted(list(vuodet))]
city_options = [{'label':s, 'value': s} for s in sorted(list(cities.index))]

vanha_ennuste_url = 'https://pxnet2.stat.fi:443/PXWeb/api/v1/fi/StatFin_Passiivi/vrm/vaenn/statfinpas_vaenn_pxt_128v_2040.px'
vanha_tk_ennustevuodet = [int(c) for c  in requests.get(vanha_ennuste_url).json()['variables'][1]['valueTexts']]


ennuste_url = "https://pxnet2.stat.fi:443/PXWeb/api/v1/fi/StatFin/vrm/vaenn/statfin_vaenn_pxt_139f.px"
tk_ennustevuodet = [int(c) for c  in requests.get(ennuste_url).json()['variables'][1]['valueTexts']]


ennuste_leikkaus = [c for c in vanha_tk_ennustevuodet if c in tk_ennustevuodet]

max_depth = None

norm_selittäjät = ['Ikä',
                   'Lähtö'
                    ]
nolla_selittäjät = ['Lähtö',
                    'Hed'
                   ]



server = Flask(__name__)
server.secret_key = os.environ.get('secret_key','secret')
app = dash.Dash(name = __name__, server = server, prevent_initial_callbacks=True)



app.title = 'Väestömetsä'

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            <script src="https://cdn.plot.ly/plotly-locale-fi-latest.js"></script>
<script>Plotly.setPlotConfig({locale: 'fi'});</script>
          {%renderer%}
          </footer>
          
            </body>
</html>
'''

#app.scripts.append_script({"external_url": "https://raw.githubusercontent.com/plotly/plotly.js/master/dist/plotly-locale-fi.js"})

#app.config['suppress_callback_exceptions']=True



def serve_layout():
        
        
        return html.Div(children = [
                    
                    html.Br(),
                    html.H1('Väestömetsä',style=dict(textAlign='center',fontSize=55, fontFamily='Arial')),
                    html.Br(),
                    html.H2('Johdanto',style=dict(textAlign='center',fontSize=26, fontFamily='Arial')),
                    html.P('Tällä sivulla testataan satunnaimetsän (Random Forest) kykyä ennustaa Suomen kaupunkien väestöä perustuen Tilastokeskuksen jakamaan avoimeen väestödataan. Satunnaismetsä on ns. ensemble -koneoppimistekniikka, jolla pyritään saavuttamaan parempi ennusteen laatu yhdistämällä useita oppimismalleja. Tässä tapauksessa nimensä mukaisesti satunnaismetsä muodostuu päätöspuista, jotka kaikki yrittävät itsenäisesti päätellä ennustettavan arvon. Päätöspuut muodostetaan satunnaisesti ja ennusteen tulos on kaikkien päätöspuiden tulosten keskiarvo. '),
                    html.Br(),
                    html.H2('Ennusteen rakenne',style=dict(textAlign='center',fontSize=26, fontFamily='Arial')),
                    html.P('Tärkeä kysymys satunnaismetsän käytössä on metsässä olevien puiden määrä, jonka saa tässä sovelluksessa itse valita. Muu ennustamiseen ja ennusteen laadun testaamiseen liittyvä parametrien valinta on myös mahdollistettu käyttäjän tehtäväksi. Käyttäjä voi valita ennustettavan kunnan sekä ennusteen pituuden. Tässä on hyvä huomioida, että ennusteen laatu olennaisesti heikkenee mitä pitemmälle ajalle ennuste tehdään. Käyttäjän on myös mahdollista testata ennusteen laatua, jättämällä haluttu osus datasta testidataksi. Tällöin ohjelma pyrkii ennustamaan valitun kunnan väestön viimeisimmiltä vuosilta. Opetusdatan määrän voi myös itse päättää. Tässä ajatuksena on, että liian kaukaa historiasta oleva data ei välttämättä ole edustavaa tulevaisuutta ennustettaessa tai tämän päivän väestöä selitettäessä. Malli on hyvin yksinkertainen ja perustuu siihen, että jokaisen vuoden tietyn ikäinen väestö selittyy iällä sekä edellisen vuoden vuotta nuorempien määrällä. Ikä sinänsä selittää kuolemanvaaraa (esim. 90 -vuotiailla on huomattavasti lyhyempi elinajanodote kuin 7 -vuotiailla) ja väestön muutosta (esim. opiskeluikäiset muuttavat toisiin kaupunkeihin opiskelumahdollisuuksien mukaan). Nollavuotiaat ennustetaan omalla satunnaismetsällään perustuen käyttäjän valitsemiin hedelmällisyysikiin. Näin voidaan määrittää minkä ikäisen väestön oletataan selittävän nollavuotiaiden määrää.' ),
                    html.Br(),
                    html.H2('Ennusteen simuloiminen ja testaaminen',style=dict(textAlign='center',fontSize=26, fontFamily='Arial')),
                    html.P('Väestön ennustamisessa ensimmäinen ennustettava vuosi ennustetaan tiedettyjen edellisen vuoden toteutuneiden faktatietojen perusteella. Seuraavan vuoden ennustamisessa hyödynnetään edellisen vuoden ennustettuja tietoja ja näin jatketaan kunnes kaikki tulevat vuodet on käyty läpi. Simulaatiossa tämä toimenpide suoritetaan menneisyydelle. Näin saadaan parempi käsitys kuinka hyvin ennustemalli toimisi pitemmälle tulevaisuuteen. Tässä tapauksessa simulaation laatua mitataan kolmella indikaattorilla (absoluuttinen keskivirhe (MAE), keskimääräinen neliövirhe (RMSE) sekä selitysaste (R²) ). Se onko ennuste luotettava indikaattorien perusteella jää käyttäjän harkitsemaksi. Tässä sovelluksessa indikaattorien hyvyyttä kuvaa suuntaa antavat liikennevalovärit (punainen, oranssi ja vihreä). Mukaan on myös otettu Tilastokeskuksen ennuste, jonka ennustetut väestömäärät on testattu niiltä vuosilta, joilta on olemassa myös Tilastokeskuksen vahvistamat väestön toteumatiedot. Tilastokeskus tekee väestöennusteen kunnittain kolmen vuoden välein. Viimeisin Tilastokeskuksen ennuste on vuodelta '+str(min(tk_ennustevuodet))+'. Tilastokeskuksen aiempi ennuste haetaan Statfinin arkistorajapinnan kautta. Näin saadaan myös viimeisimmän TK:n ennustetta edeltävät kaksi ennustevuotta vertailuun mukaan. Testaus suoritetaan samoilla indikaattoreilla kuin simulaatiossa. Testaaminen eroaa simulaatiosta siten, että testaamisessa vain vertaillaan saatuja arvoja riippumatta arvojen muodostustavasta. Sekä Tilastokeskuksen, että tämän koneoppimismallin testitiedot saa vietyä, muiden metatietojen ohella, Excel-tiedostoon sivun alalaitaan ilmestyvän linkin kautta. Itse väestöennusteprojektio luodaan automaattisesti testiajojen jälkeen.'),
                    html.Br(),
                    html.H2('Lopuksi',style=dict(textAlign='center',fontSize=26, fontFamily='Arial')),
                    html.P('Tämä aplikaatio on luotu tarkoituksena tutustuttaa käyttäjä koneoppimisen ihmeelliseen maailmaan. Toivotan siis iloisia hetkiä väestön ennustamisen ja ennakoivan analytiikan sekä tämän aplikaation käytön aikana. Lisätietoja käytetyistä tekniikoista ja testausmääreistä löydät tämän sivun alaosasta.'),
        html.Br(),
                    html.Div(className = 'row',
                             children=[

                                        html.Div(className='four columns',children=[
                                                html.H2('Valitse ennusteen pituus.'),
                                                dcc.Slider(id='pituus',
                                                           min=1,
                                                           max=50,
                                                           step=1,
                                                           value=10,
                                                           marks = {
                                                           1: '1 vuosi',
                                                           20: '20 vuotta',
                                                           
                                                           50: '50 vuotta'},
                                                           updatemode='drag'
                                                          ),
                                                html.Br(),
                                                html.Div(id='pituus_indicator', style={'margin-top': 20})
                                            
                                                ]),
                                 html.Div(className='four columns',children=[
                                                html.H2('Valitse puiden määrä metsässä.'),
                                                dcc.Slider(id='puut',
                                                           min = 100,
                                                           max = 500,
                                                           step=1,
                                                           value=100,
                                                           marks = {
                                                           100: '100 puuta',
                                                           500: '500 puuta'
                                                          # 1000: '1000 puuta'
                                                           },
                                                           updatemode='drag'
                                                          ),
                                                 html.Br(),
                                                 html.Div(id = 'tree_indicator', style={'margin-top': 20})
                                                ]),
                                  html.Div(className='four columns',children=[
                                                html.H2('Valitse hedelmällisyysiät.'),
                                                dcc.RangeSlider(id='hed',
                                                           min = 15,
                                                           max = 50,
                                                           step=1,
                                                           value=[20, 35],
                                                           marks = {
                                                           15: '15 -vuotiaat',
                                                           
                                                           30: '30 -vuotiaat',
                                                           50: '50 -vuotiaat'
                                                           },
                                                           updatemode='drag'
                                                          ),
                                                 html.Br(),
                                                 html.Div(id = 'fertility_indicator', style={'margin-top': 20})
                                                ])
                             ]
                    ),
                   html.Br(),
                   html.Div(className = 'row',
                             children=[
                                 
                                 
                             html.Div(className='four columns',children=[
                                            
                                                 html.H2('Valitse opetuksen aloitusvuosi.'),
                                                 dcc.Slider(id='alkuvuosi',
                                                           min=max(vuodet)-30,
                                                           max=max(vuodet)-2,
                                                           step=1,
                                                           value=2010,
                                                           marks = {
                                                           max(vuodet)-30: str(max(vuodet)-30),

                                                           max(vuodet)-2: str(max(vuodet)-2)
                                                           },
                                                           updatemode='drag'
                                                          ),
                                                 html.Br(),
                                                 html.Div(id = 'year_indicator', style={'margin-top': 20})
                                                ]),
                                 
                                 
                                        html.Div(className='four columns',children=[
                                            
                                                 html.H2('Valitse testikoko.'),
                                                 dcc.Slider(id='testikoko',
                                                           min=10,
                                                           max=30,
                                                           step=1,
                                                           value=20,
                                                           marks = {
                                                           10: '10 %',
                                                           20: '20 %',
                                                           30: '30 %'},
                                                           updatemode='drag'
                                                          ),
                                                 html.Br(),
                                                 html.Div(id = 'test_indicator', style={'margin-top': 20})
                                                ]),
                                         html.Div(className='four columns', children =[
                                             
                                             html.H2('Valitse kunta.'),
                                             dcc.Dropdown(id = 'kunnat',
                                                              multi=False,
                                                              options = city_options,
                                                              value='Helsinki')
                                         ]
                                                 )
                                        
                             ]
                                                  
                    ),
                    html.Br(),
                    html.Button('Testaa ja ennusta.', id='launch', n_clicks=0),
                    
                    html.Br(),
                    html.Br(),
                    dcc.Loading(id='spinner',fullscreen=False, type=spinner_type, children=[html.Div(id='ennuste')]),

                    html.Label(['Datan lähde: ', 
                                html.A('StatFin', href = "https://pxnet2.stat.fi/PXWeb/pxweb/fi/StatFin/StatFin__vrm__vaerak/statfin_vaerak_pxt_11re.px/")
                               ]),
                    html.Label(['Tilastokeskuksen väestöennuste: ',
                               html.A('StatFin', href = "https://pxnet2.stat.fi/PXWeb/pxweb/fi/StatFin/StatFin__vrm__vaenn/statfin_vaenn_pxt_128v.px/")
                               ]),
                    html.Label(['Satunnaismetsä Wikipediassa: ', 
                                html.A('Wikipedia', href='https://en.wikipedia.org/wiki/Random_forest')
                               ]),
                    html.Label(['Regressiometriikoista : ', 
                                html.A('Medium', href='https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914')
                               ]),
                    html.Label(['Sovellus GitHubissa: ', 
                                html.A('GitHub', href='https://github.com/tuopouk/vaestometsa/tree/master')
                               ]),
                    html.Label(['Väestömetsä voitti Datamenestyjät 2021 kilpailun!: ', 
                                html.A('Tilastokeskus', href='https://www.stat.fi/uutinen/vaestometsa-palvelu-voitti-ensimmaisen-datamenestyjat-kilpailun')
                               ]),
                    html.Label(['Väestömetsä lyhyt esittely Youtubessa: ', 
                                html.A('Youtube', href='https://www.youtube.com/watch?v=qy5d6vca3n0')
                               ]),
            
                    html.Label(['Tehnyt Tuomas Poukkula. ', 
                                html.A('Seuraa Twitterissä.', href='https://twitter.com/TuomasPoukkula')
                               ]),
                    html.Label(['Seuraa myös LinkedIn:ssä. ', 
                                html.A('LinkedIn', href='https://www.linkedin.com/in/tuomaspoukkula/')
                               ]),
                    
        
        
    ])

    
@app.callback(
    Output('pituus_indicator', 'children'),
    [Input('pituus', 'value')])
def update_pituus(value):
    
    return 'Valittu ennusteen pituus: {} vuotta'.format(
        str(value)
    )    
    
@app.callback(
    Output('tree_indicator', 'children'),
    [Input('puut', 'value')])
def update_puut(value):
    
    return 'Puita metsässä: {} puuta'.format(
        str(value)
    )  
    
@app.callback(
    Output('fertility_indicator', 'children'),
    [Input('hed', 'value')])
def update_hed(value):
    
    return 'Valitut hedelmällisyysiät: {}  - vuotiaat'.format(
        str(min(value))+' - '+str(max(value))
    )  

@app.callback(
    Output('year_indicator', 'children'),
    [Input('alkuvuosi', 'value')])
def update_year(value):
    
    return 'Valittu opetuksen aloitusvuosi: {} '.format(
        str(value)
    )

# @app.callback(
#     Output('year_selection_indicator', 'children'),
#     [Input('vuosivalitsin', 'value')])
# def update_vuosivalitsin(value):
    
#     return 'Väestöennuste iän mukaan vuodelle {} '.format(
#         str(value)
#     )  

@app.callback(
    Output('test_indicator', 'children'),
    [Input('testikoko', 'value')])
def update_test(value):
    
    return 'Valittu testikoko: {} '.format(
        str(value)+' %'
    )  




def apply_uncertainty(year, first_predicted):
    
    if year < first_predicted:
        return 'Toteutunut'
    if year in range(first_predicted, first_predicted+11):
        return 'Ennuste'
    if year in range(first_predicted+11,first_predicted+21):
        return 'Epävarma ennuste'
    if year > first_predicted + 20:
        return 'Erittäin epävarma ennuste'
        

    


def get_new_tk_forecast(city_code):
    
    ennuste_url = 'https://pxnet2.stat.fi:443/PXWeb/api/v1/fi/StatFin/vrm/vaenn/statfin_vaenn_pxt_139f.px'
    
    headers =  {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                                'Content-Type':'application/json'
                               }
    
    ennuste_query = {
                          "query": [
                            {
                              "code": "Alue",
                              "selection": {
                                "filter": "item",
                                "values": [
                                    city_code
                                ]
                              }
                            },
                            {
                              "code": "Ikä",
                              "selection": {
                                "filter": "item",
                                "values": [
                                  "000",
                                  "001",
                                  "002",
                                  "003",
                                  "004",
                                  "005",
                                  "006",
                                  "007",
                                  "008",
                                  "009",
                                  "010",
                                  "011",
                                  "012",
                                  "013",
                                  "014",
                                  "015",
                                  "016",
                                  "017",
                                  "018",
                                  "019",
                                  "020",
                                  "021",
                                  "022",
                                  "023",
                                  "024",
                                  "025",
                                  "026",
                                  "027",
                                  "028",
                                  "029",
                                  "030",
                                  "031",
                                  "032",
                                  "033",
                                  "034",
                                  "035",
                                  "036",
                                  "037",
                                  "038",
                                  "039",
                                  "040",
                                  "041",
                                  "042",
                                  "043",
                                  "044",
                                  "045",
                                  "046",
                                  "047",
                                  "048",
                                  "049",
                                  "050",
                                  "051",
                                  "052",
                                  "053",
                                  "054",
                                  "055",
                                  "056",
                                  "057",
                                  "058",
                                  "059",
                                  "060",
                                  "061",
                                  "062",
                                  "063",
                                  "064",
                                  "065",
                                  "066",
                                  "067",
                                  "068",
                                  "069",
                                  "070",
                                  "071",
                                  "072",
                                  "073",
                                  "074",
                                  "075",
                                  "076",
                                  "077",
                                  "078",
                                  "079",
                                  "080",
                                  "081",
                                  "082",
                                  "083",
                                  "084",
                                  "085",
                                  "086",
                                  "087",
                                  "088",
                                  "089",
                                  "090",
                                  "091",
                                  "092",
                                  "093",
                                  "094",
                                  "095",
                                  "096",
                                  "097",
                                  "098",
                                  "099",
                                  "100-"
                                ]
                              }
                            }
                          ],
                          "response": {
                            "format": "json-stat2"
                          }
                        }
    
    blocked = True
    while blocked:
        try:
            ennuste_json = requests.post(ennuste_url,data=json.dumps(ennuste_query),headers=headers)
            tk_data = ennuste_json.json()
            blocked=False
        except:
                
            time.sleep(2)
    
    
    tk_year_df = pd.DataFrame()
    tk_age_df = pd.DataFrame()
    tk_year_df['Vuosi'] = [int(c) for c in tk_data['dimension']['Vuosi']['category']['label'].values()]
    tk_year_df['index']=0
    tk_age_df['Ikä'] = [int(c) for c in tk_data['dimension']['Ikä']['category']['index'].values()]
    tk_age_df['index'] = 0
    tk_data_df = pd.merge(left=tk_age_df,right=tk_year_df,how='outer',on='index').drop_duplicates().sort_values(by=['Vuosi','Ikä'])[['Vuosi','Ikä']].rename(columns={'ikä':'Ikä'})
    tk_data_df['Tilastokeskuksen ennuste']  = tk_data['value']
    tk_data_df = tk_data_df.set_index('Vuosi')
    tk_data_df['Kaupunki'] = list(tk_data['dimension']['Alue']['category']['label'].values())[0].capitalize()
    tk_data_df = tk_data_df[['Kaupunki','Ikä','Tilastokeskuksen ennuste']]
    
    return tk_data_df

def get_old_tk_forecast(city_code):
    
    ennuste_url = 'https://pxnet2.stat.fi:443/PXWeb/api/v1/fi/StatFin_Passiivi/vrm/vaenn/statfinpas_vaenn_pxt_128v_2040.px'
    
    headers =  {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                                'Content-Type':'application/json'
                               }
    
    
    ennuste_query = {
                          "query": [
                            {
                              "code": "Alue",
                              "selection": {
                                "filter": "item",
                                "values": [
                                    city_code
                                ]
                              }
                            },
                            {
                              "code": "Ikä",
                              "selection": {
                                "filter": "item",
                                "values": [
                                  "000",
                                  "001",
                                  "002",
                                  "003",
                                  "004",
                                  "005",
                                  "006",
                                  "007",
                                  "008",
                                  "009",
                                  "010",
                                  "011",
                                  "012",
                                  "013",
                                  "014",
                                  "015",
                                  "016",
                                  "017",
                                  "018",
                                  "019",
                                  "020",
                                  "021",
                                  "022",
                                  "023",
                                  "024",
                                  "025",
                                  "026",
                                  "027",
                                  "028",
                                  "029",
                                  "030",
                                  "031",
                                  "032",
                                  "033",
                                  "034",
                                  "035",
                                  "036",
                                  "037",
                                  "038",
                                  "039",
                                  "040",
                                  "041",
                                  "042",
                                  "043",
                                  "044",
                                  "045",
                                  "046",
                                  "047",
                                  "048",
                                  "049",
                                  "050",
                                  "051",
                                  "052",
                                  "053",
                                  "054",
                                  "055",
                                  "056",
                                  "057",
                                  "058",
                                  "059",
                                  "060",
                                  "061",
                                  "062",
                                  "063",
                                  "064",
                                  "065",
                                  "066",
                                  "067",
                                  "068",
                                  "069",
                                  "070",
                                  "071",
                                  "072",
                                  "073",
                                  "074",
                                  "075",
                                  "076",
                                  "077",
                                  "078",
                                  "079",
                                  "080",
                                  "081",
                                  "082",
                                  "083",
                                  "084",
                                  "085",
                                  "086",
                                  "087",
                                  "088",
                                  "089",
                                  "090",
                                  "091",
                                  "092",
                                  "093",
                                  "094",
                                  "095",
                                  "096",
                                  "097",
                                  "098",
                                  "099",
                                  "100-"
                                ]
                              }
                            }
                          ],
                          "response": {
                            "format": "json-stat2"
                          }
                        }
    
    blocked = True
    while blocked:
        try:
            ennuste_json = requests.post(ennuste_url,data=json.dumps(ennuste_query),headers=headers)
            tk_data = ennuste_json.json()
            blocked = False
        except:
                
            time.sleep(2)
        
    
    tk_year_df = pd.DataFrame()
    tk_age_df = pd.DataFrame()
    tk_year_df['Vuosi'] = [int(c) for c in tk_data['dimension']['Vuosi']['category']['label'].values()]
    tk_year_df['index'] = 0
    tk_age_df['Ikä'] = [int(c) for c in tk_data['dimension']['Ikä']['category']['index'].values()]
    tk_age_df['index'] = 0
    tk_data_df = pd.merge(left=tk_age_df,right=tk_year_df,how='outer',on='index').drop_duplicates().sort_values(by=['Vuosi','Ikä'])[['Vuosi','Ikä']].rename(columns={'ikä':'Ikä'})
    tk_data_df['Tilastokeskuksen ennuste']  = tk_data['value']
    tk_data_df = tk_data_df.set_index('Vuosi')
    tk_data_df['Kaupunki'] = list(tk_data['dimension']['Alue']['category']['label'].values())[0].capitalize()
    tk_data_df = tk_data_df[['Kaupunki','Ikä','Tilastokeskuksen ennuste']]
    
    return tk_data_df

def get_data(city_code):
    
    url = 'https://pxnet2.stat.fi:443/PXWeb/api/v1/fi/StatFin/vrm/vaerak/statfin_vaerak_pxt_11re.px'
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                                'Content-Type':'application/json'
                               }
    payload = {
          "query": [
            {
              "code": "Alue",
              "selection": {
                "filter": "item",
                "values": [
                  city_code
                ]
              }
            },
            {
              "code": "Ikä",
              "selection": {
                "filter": "item",
                "values": [
                  "000",
                  "001",
                  "002",
                  "003",
                  "004",
                  "005",
                  "006",
                  "007",
                  "008",
                  "009",
                  "010",
                  "011",
                  "012",
                  "013",
                  "014",
                  "015",
                  "016",
                  "017",
                  "018",
                  "019",
                  "020",
                  "021",
                  "022",
                  "023",
                  "024",
                  "025",
                  "026",
                  "027",
                  "028",
                  "029",
                  "030",
                  "031",
                  "032",
                  "033",
                  "034",
                  "035",
                  "036",
                  "037",
                  "038",
                  "039",
                  "040",
                  "041",
                  "042",
                  "043",
                  "044",
                  "045",
                  "046",
                  "047",
                  "048",
                  "049",
                  "050",
                  "051",
                  "052",
                  "053",
                  "054",
                  "055",
                  "056",
                  "057",
                  "058",
                  "059",
                  "060",
                  "061",
                  "062",
                  "063",
                  "064",
                  "065",
                  "066",
                  "067",
                  "068",
                  "069",
                  "070",
                  "071",
                  "072",
                  "073",
                  "074",
                  "075",
                  "076",
                  "077",
                  "078",
                  "079",
                  "080",
                  "081",
                  "082",
                  "083",
                  "084",
                  "085",
                  "086",
                  "087",
                  "088",
                  "089",
                  "090",
                  "091",
                  "092",
                  "093",
                  "094",
                  "095",
                  "096",
                  "097",
                  "098",
                  "099",
                  "100-"
                ]
              }
            }
          ],
          "response": {
            "format": "json-stat"
          }
        }
    blocked = True
    while blocked:
        try:
            data_json = requests.post(url,data=json.dumps(payload),headers=headers)
            data = data_json.json()
            blocked=False
        except:
                
            time.sleep(2)
    
    age_df = pd.DataFrame()
    age_df['Ikä'] = list(data['dataset']['dimension']['Ikä']['category']['index'].values())
    year_df = pd.DataFrame()
    year_df['Vuosi'] = [int(c) for c in list(data['dataset']['dimension']['Vuosi']['category']['label'].values())]
    year_df['index']=0
    age_df['index']=0
    data_df = pd.merge(left=age_df,right=year_df,how='outer',on='index').drop_duplicates().sort_values(by=['Ikä','Vuosi'])[['Vuosi','Ikä']]
    data_df['Väestö'] = data['dataset']['value']
    data_df['Kaupunki'] = list(data['dataset']['dimension']['Alue']['category']['label'].values())[0].capitalize()
    data_df = data_df.set_index('Vuosi')
    
    
    return data_df

def preprocess(data_df, hed_min, hed_max):
    
    iät = sorted(list(pd.unique(data_df.Ikä)))
    
    nollat_prev = data_df[data_df.Ikä==0] 
    nollat_prev = nollat_prev.loc[nollat_prev.index < nollat_prev.index.max()]

    nollat = data_df[data_df.Ikä==0] 
    nollat = nollat.loc[nollat.index > nollat.index.min()]

    nollat['Lähtö'] = nollat_prev[['Väestö']].values
    nollat_last = nollat.iloc[-1:,:].copy()
    nollat_last.index +=1
    nollat_last.Lähtö = nollat_last.Väestö
    nollat_last.Väestö=np.nan
    nollat=pd.concat([nollat,nollat_last],axis=0)

    hed = data_df[data_df.Ikä.isin(np.arange(hed_min,hed_max+1))].reset_index().groupby('Vuosi').agg({'Väestö':'sum'}).rename(columns={'Väestö':'Hed'})

    hed_prev = hed[hed.index < hed.index.max()]
    hed = hed[hed.index > hed.index.min()]
    hed['Lähtö'] = hed_prev[['Hed']].values

    hed_last = hed.iloc[-1:,:].copy()
    hed_last.index += 1
    hed_last.Lähtö = hed_last.Hed
    hed_last.Hed = np.nan
    hed = pd.concat([hed,hed_last],axis=0)
    hed.drop('Hed',axis=1,inplace=True)
    hed = hed.rename(columns={'Lähtö':'Hed'})


    nollat = pd.merge(left=nollat,right=hed, how='left', left_on=nollat.index, right_on=hed.index).rename(columns={'key_0':'Vuosi'}).set_index('Vuosi')

    nollat['Muutos'] = nollat['Väestö'] - nollat['Lähtö']
    nollat = nollat.rename(columns={'Väestö':'Ennusta'})

    df_0_99 = data_df[data_df.Ikä.isin(np.arange(min(iät),max(iät)))].copy()
    df_0_99 = df_0_99.loc[df_0_99.index < df_0_99.index.max()]

    df_1_100 = data_df[data_df.Ikä.isin(np.arange(min(iät)+1,max(iät)+1))]
    df_1_100 = df_1_100.loc[df_1_100.index > df_1_100.index.min()]

    df_1_100['Lähtö'] = df_0_99[['Väestö']].values


    df_1_100_last = df_1_100.loc[df_1_100.index == df_1_100.index.max()].copy()
    df_1_100_last.index += 1
    df_1_100_last.Lähtö = df_1_100_last.Väestö
    df_1_100_last.Väestö = np.nan
    df_1_100_last = pd.concat([nollat_last,df_1_100_last])
    df_1_100_last.Ikä+=1
    df_1_100_last = df_1_100_last[df_1_100_last.Ikä<=max(iät)]

    väestö = pd.concat([df_1_100,df_1_100_last],axis=0)
    väestö['Kohorttimuutos'] = väestö['Väestö'] - väestö['Lähtö']
    väestö = väestö.rename(columns={'Väestö':'Ennusta'})

    nollat=nollat.reset_index()
    väestö=väestö.reset_index()
    return [nollat, väestö]


@app.callback(
    Output('ennuste','children'),
    [

    Input('launch', 'n_clicks')
    ],
    [
    State('pituus','value'),
    State('puut','value'),
    State('alkuvuosi','value'),
    State('testikoko','value'),
    State('hed','value'),
    State('kunnat','value')
    
    ]
)


def test_predict_document(n_clicks,ennusteen_pituus, puut, aloita, testikoko, hed, city):
    
#     global prediction_result
#     global tk_result
#     global ennusteen_pituus

    


    if n_clicks > 0:
        
        
        
        clicked = True
                               

        test_size = testikoko/100
       

        hed_min = min(hed)
        hed_max = max(hed)
        
        city_code = cities.loc[city.strip().capitalize()].aluekoodi
        
        
        
        data_df = get_data(city_code)
        data_df = data_df.loc[data_df.index>=aloita]
        
        testivuodet = int(math.ceil((data_df.index.max() - aloita)*test_size))
        
        

        testi_alkuvuosi = data_df.index.max()-testivuodet
        
        testattavat = ', '.join([str(testi_alkuvuosi+i) for i in range(testivuodet+1)])
        
        alkuvuosi = data_df.index.max()
        
        
        
        svr = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=max_depth,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=2, min_samples_split=30,
                      min_weight_fraction_leaf=0.0, n_estimators=puut,
                      n_jobs=-1, oob_score=True, random_state=9876, verbose=0,
                      warm_start=False)


        ridge = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=max_depth,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=2, min_samples_split=30,
                      min_weight_fraction_leaf=0.0, n_estimators=puut,
                      n_jobs=-1, oob_score=True, random_state=9876, verbose=0,
                      warm_start=False)



        # Mähläys

        
        # 1. Baseline (Tilastokeskuksen ennuste)
        
        old_tk_forecast = get_old_tk_forecast(city_code)
       
        new_tk_forecast = get_new_tk_forecast(city_code)

        ennuste_leikkaus = [i for i in pd.unique(old_tk_forecast.index) if i not in pd.unique(new_tk_forecast.index)]
        toteutuneet_ennusteet = len(ennuste_leikkaus)

        
        tk_forecast = pd.concat([old_tk_forecast.loc[ennuste_leikkaus],new_tk_forecast],axis=0).sort_index()
        
        
 

        tk_test = pd.merge(left=data_df.drop('Kaupunki',axis=1), right=tk_forecast, how='inner', left_on=[data_df.index,'Ikä'], right_on=[tk_forecast.index,'Ikä']).rename(columns={'key_0':'Vuosi'}).sort_values(by=['Vuosi','Ikä']).set_index('Vuosi')
        
        try:

            nmae_tk = round(mean_absolute_error(tk_test.Väestö,tk_test['Tilastokeskuksen ennuste'])/tk_test.Väestö.std(),2)
            nrmse_tk = round(math.sqrt(mean_squared_error(tk_test.Väestö,tk_test['Tilastokeskuksen ennuste']))/tk_test.Väestö.std(),2)
            r2_tk = round(r2_score(tk_test.Väestö,tk_test['Tilastokeskuksen ennuste']),2)

            v_tot = int(tk_test.loc[tk_test.index.max()].Väestö.sum())
            v_enn = int(tk_test.loc[tk_test.index.max()]['Tilastokeskuksen ennuste'].sum())
            diff = v_enn-v_tot
            v_diff_document = diff

            v_tot_document = v_tot
            v_enn_document = v_enn
            
            if diff > 0:
                diff_word = 'erosi toteutuneesta: '+str(diff)+' henkilöä enemmän.'
            elif diff < 0:
                diff_word = 'erosi toteutuneesta: '+str(np.absolute(diff))+' henkilöä vähemmän.'
            else:
                diff_word = 'ennusti täsmälleen saman väestön.'
            
            
            
            tk_chain = 'Tilastokeskuksen ennusteen vastaavat arvot MAE: '+str(nmae_tk)+', RMSE: '+str(nrmse_tk)+', R²: '+str(r2_tk)+'.'
            #tk_chain+= '. Toteutunut väestö vuodelle '+str(tk_test[tk_test.vuosi==tk_test.vuosi.max()].vuosi)+': '+'{:,}'.format()
            tk_chain+= ' Tilastokeskuksen ennuste: '+'{:,}'.format(v_enn).replace(',',' ')
            tk_chain += '. Tilastokeskuksen ennuste '+diff_word
        except:
            # Ei voi testata.
            tk_chain = ''

       # Nollat ja 1-100 mähläys 


        
        nollat, väestö = preprocess(data_df,hed_min,hed_max)

#         nollat=nollat.reset_index()
#         väestö=väestö.reset_index()



        # Testi
        
        
        
        # Simulaatio: opetus ensimmäiseen testivuoteen asti, minkä jälkeen projektiota jatketaan kuten normaalisti. Lopuksi mitataan simulaation laatu.
        
        

        scl = StandardScaler()
        scl2 = StandardScaler()

        väestö_ = väestö.copy()

        nollat_=nollat.copy()


        x = väestö_[väestö_.Vuosi<testi_alkuvuosi][norm_selittäjät]
        X = scl.fit_transform(x)
        y= väestö_[väestö_.Vuosi<testi_alkuvuosi]['Kohorttimuutos']

        ridge.fit(X,y)

        x = nollat_[nollat_.Vuosi<testi_alkuvuosi][nolla_selittäjät]
        X = scl2.fit_transform(x)
        y = nollat_[nollat_.Vuosi<testi_alkuvuosi]['Muutos']

        svr.fit(X,y)

        v = väestö_.copy()

        v_20 = v[v.Vuosi==testi_alkuvuosi]
        v = v[v.Vuosi<testi_alkuvuosi]

        v_20.Kohorttimuutos =  ridge.predict(scl.transform(v_20[norm_selittäjät]))
        v_20.Ennusta = np.maximum(0,v_20.Lähtö + v_20.Kohorttimuutos)
        v = pd.concat([v,v_20],axis=0)


        n = nollat_.copy()


        n_20 = n[n.Vuosi==testi_alkuvuosi]
        n = n[n.Vuosi<testi_alkuvuosi]

        n_20.Muutos =  svr.predict(scl2.transform(n_20[nolla_selittäjät]))
        n_20.Ennusta = np.maximum(0,n_20.Lähtö + n_20.Muutos)
        n = pd.concat([n,n_20],axis=0)


        for year in tqdm(range(testi_alkuvuosi+1, alkuvuosi+1)):

            hed_df = v[(v.Vuosi == year -1) & (v.Ikä.isin(np.arange(hed_min,hed_max+1)))].groupby('Vuosi').agg({'Ennusta':'sum'}).rename(columns={'Ennusta':'Hed'}).reset_index().copy()


            nolla_df = n[(n.Vuosi==year-1)].copy()

            ykköset = nolla_df.copy()

            ykköset.Lähtö = ykköset.Ennusta
            ykköset.Vuosi+=1
            ykköset.Ikä+=1
            ykköset = ykköset[['Vuosi',

                               'Ikä','Lähtö']]


            nolla_df.Lähtö = nolla_df.Ennusta
            nolla_df.Vuosi+=1



            nolla_df['Hed']=hed_df['Hed'].values


            loput = v[(v.Vuosi==year-1)&(v.Ikä<100)].copy()
            loput.Ikä+=1
            loput.Vuosi+=1
            loput.Lähtö=loput.Ennusta
            loput.drop(['Ennusta','Kohorttimuutos'],axis=1, inplace=True)

            loput = pd.concat([ykköset,loput],axis=0)


            nolla_df['Muutos'] = svr.predict(scl2.transform(nolla_df[nolla_selittäjät]))
            nolla_df['Ennusta'] = np.maximum(0, nolla_df.Lähtö + nolla_df.Muutos)

            n = pd.concat([n,nolla_df], axis = 0)



            loput['Kohorttimuutos'] = ridge.predict(scl.transform(loput[norm_selittäjät]))
            loput['Ennusta'] = np.maximum(0,loput.Lähtö + loput.Kohorttimuutos)

            v = pd.concat([v,loput],axis = 0)

        result = pd.concat([n[['Vuosi',
                               'Ikä',

                               'Ennusta']],v[['Vuosi',
                                              'Ikä',

                                              'Ennusta']]],axis = 0)

        test_result=result.sort_values(by='Ikä')
        toteutunut = pd.concat([nollat_[(nollat_.Vuosi<=alkuvuosi)][['Vuosi','Ikä','Ennusta']],väestö_[(väestö_.Vuosi<=alkuvuosi)][['Vuosi','Ikä','Ennusta']]],axis=0)
        test_toteutunut = toteutunut.sort_values(by='Ikä')
        
        simulation_test = test_result.copy().set_index('Vuosi')
        simulation_true = test_toteutunut.copy().set_index('Vuosi')
        
        
        mae = mean_absolute_error(test_toteutunut[test_toteutunut.Vuosi>=testi_alkuvuosi].Ennusta, 
                                  test_result[test_result.Vuosi>=testi_alkuvuosi].Ennusta)
        margin= 1.96*(test_toteutunut[test_toteutunut.Vuosi>=testi_alkuvuosi].Ennusta- test_result[test_result.Vuosi>=testi_alkuvuosi].Ennusta).std()/math.sqrt(len(test_result[test_result.Vuosi>=testi_alkuvuosi]))
        
        nmae = round(mae / test_toteutunut[test_toteutunut.Vuosi>=testi_alkuvuosi].Ennusta.std(),2)
        
        rmse = math.sqrt(mean_squared_error(test_toteutunut[test_toteutunut.Vuosi>=testi_alkuvuosi].Ennusta, 
                                           test_result[test_result.Vuosi>=testi_alkuvuosi].Ennusta))
        
        nrmse = round(rmse / test_toteutunut[test_toteutunut.Vuosi>=testi_alkuvuosi].Ennusta.std(),2)
        
        r2 = round(r2_score(test_toteutunut[test_toteutunut.Vuosi>=testi_alkuvuosi].Ennusta, 
                       test_result[test_result.Vuosi>=testi_alkuvuosi].Ennusta),2)
        
        if nmae <= 0.1:
            nmae_color = 'green'
            nmae_text = 'Hyvä'
        elif nmae > 0.1 and nmae < 0.5:
            nmae_color = 'orange'
            nmae_text = 'Kohtalainen'
        else:
            nmae_color = 'red'
            nmae_text = 'Huono'
            
        if nrmse <= 0.2:
            nrmse_color = 'green'
            nrmse_text = 'Hyvä'
        elif nrmse > 0.2 and nrmse < 0.6:
            nrmse_color = 'orange'
            nrmse_text='Kohtalainen'
        else:
            nrmse_color = 'red'
            nrmse_text = 'Huono'
            
        
        if r2 <= 0.5:
            r2_color = 'red'
            r2_text = 'Huono'
        elif r2 > 0.5 and r2 < 0.8:
            r2_color = 'orange'
            r2_text = 'Kohtalainen'
        else:
            r2_color = 'green'
            r2_text='Hyvä'
        

#         chain = 'MAE: '
#         chain+= str(round(mae,2))
       # chain+=', Virhemarginaali: '
       # chain+=str(round(margin,2))
        chain=''
        chain+='NMAE: '
        chain+= str(nmae)
#         chain+=', RMSE: '
#         chain+=str(round(rmse,2))
        chain+=', NRMSE: '
        chain+= str(nrmse)
        chain+=', R²: '
        chain+=str(r2)


        
        # Testi: ennustetaan viimeisimpien vuosien väestö ja vertaillaan Tilastokeskuksen ennusteeseen (ne vuodet, jotka ovat toteutuneissa vuosissa ja Tilastokeskuksen ennusteessa).
        
        
        if toteutuneet_ennusteet > 0:
                       
                        
            scl = StandardScaler()
            scl2 = StandardScaler()

            väestö_ = väestö.copy()
            nollat_=nollat.copy()
            
            vika_vuosi = sorted(ennuste_leikkaus)[0]

            
            x = väestö_[väestö_.Vuosi<vika_vuosi][norm_selittäjät]
            X = scl.fit_transform(x)
            y= väestö_[väestö_.Vuosi<vika_vuosi]['Kohorttimuutos']

            ridge.fit(X,y)

            x = nollat_[nollat_.Vuosi<vika_vuosi][nolla_selittäjät]
            X = scl2.fit_transform(x)
            y = nollat_[nollat_.Vuosi<vika_vuosi]['Muutos']

            svr.fit(X,y)

            v = väestö_.copy()

            v_20 = v[(v.Vuosi.isin(ennuste_leikkaus))]
            
            v = v[v.Vuosi<vika_vuosi]

            v_20.Kohorttimuutos =  ridge.predict(scl.transform(v_20[norm_selittäjät]))
            v_20.Ennusta = np.maximum(0,v_20.Lähtö + v_20.Kohorttimuutos)
            v = pd.concat([v,v_20],axis=0)


            n = nollat_.copy()


            n_20 = n[(n.Vuosi.isin(ennuste_leikkaus))]
            n = n[n.Vuosi<vika_vuosi]

            n_20.Muutos =  svr.predict(scl2.transform(n_20[nolla_selittäjät]))
            n_20.ennusta = np.maximum(0,n_20.Lähtö + n_20.Muutos)
            n = pd.concat([n,n_20],axis=0)

                         


            result = pd.concat([n[['Vuosi',
                                   'Ikä',

                                   'Ennusta']],v[['Vuosi',
                                                  'Ikä',

                                                  'Ennusta']]],axis = 0)

            quick_test_result=result.sort_values(by='Ikä')
            toteutunut = pd.concat([nollat[(nollat.Vuosi.isin(ennuste_leikkaus))][['Vuosi','Ikä','Ennusta']],väestö[(väestö.Vuosi.isin(ennuste_leikkaus))][['Vuosi','Ikä','Ennusta']]],axis=0)

    
            quick_test_toteutunut = toteutunut.sort_values(by='Ikä')
        
            qtr = quick_test_result[quick_test_result.Vuosi.isin(ennuste_leikkaus)]
            qtt = quick_test_toteutunut[quick_test_toteutunut.Vuosi.isin(ennuste_leikkaus)]
            
            
            age_test = qtr[qtr.Vuosi==vika_vuosi]
            age_true = qtt[qtt.Vuosi==vika_vuosi]
            
            
            quick_mae = mean_absolute_error(qtt.Ennusta, qtr.Ennusta)
    #         margin= 1.96*(test_toteutunut[test_toteutunut.vuosi>=testi_alkuvuosi].ennusta- test_result[test_result.vuosi>=testi_alkuvuosi].ennusta).std()/math.sqrt(len(test_result[test_result.vuosi>=testi_alkuvuosi]))

            quick_nmae = round(quick_mae / qtt.Ennusta.std(),2)

            quick_rmse = math.sqrt(mean_squared_error(qtt.Ennusta, 
                                               qtr.Ennusta))

            quick_nrmse = round(quick_rmse / qtt.Ennusta.std(),2)

            quick_r2 = round(r2_score(qtt.Ennusta, 
                          qtr.Ennusta),2)
            
            tot_väestö = '{:,}'.format(int(qtt[qtt.Vuosi==sorted(ennuste_leikkaus)[-1]].Ennusta.sum())).replace(',',' ')
            enn_väestö = '{:,}'.format(int(np.ceil(qtr[qtr.Vuosi==sorted(ennuste_leikkaus)[-1]].Ennusta.sum()))).replace(',',' ')
            diff = int(np.ceil(qtr.Ennusta.sum()))-int(qtt.Ennusta.sum())
            diff_document = diff
            
            enn_document = int(np.ceil(qtr[qtr.Vuosi==sorted(ennuste_leikkaus)[-1]].Ennusta.sum()))
            
            if toteutuneet_ennusteet == 1:
                printed_value=''
            else:
                printed_value = str(number_to[toteutuneet_ennusteet])
            
            if diff > 0:
                diff_word = 'erosi toteutuneesta: '+str(diff)+' henkilöä enemmän.'
            elif diff < 0:
                diff_word = 'erosi toteutuneesta: '+str(np.absolute(diff))+' henkilöä vähemmän.'
            else:
                diff_word = 'ennusti täsmälleen saman väestön.'
            
            try:
                quick_chain='Testitulokset '+printed_value+' viimeisimmälle vuodelle: '
                quick_chain+='MAE: '
                quick_chain+= str(quick_nmae)
                quick_chain+=', RMSE: '
                quick_chain+= str(quick_nrmse)
                quick_chain+=', R²: '
                quick_chain+=str(quick_r2)
                quick_chain+= '. Toteutunut väestö vuodelle '+str(sorted(ennuste_leikkaus)[-1])+': '+str(tot_väestö)
                quick_chain+= '. Ennustettu väestö vuodelle '+str(sorted(ennuste_leikkaus)[-1])+': '+str(enn_väestö)
                quick_chain += ', Ennuste '+diff_word
            except:
                quick_chain = ''
                

        # Projektio


        scl = StandardScaler()
        scl2 = StandardScaler()





        väestö_ = väestö.copy()
        nollat_=nollat.copy()



        x = väestö_[väestö_.Vuosi<alkuvuosi][norm_selittäjät]
        X = scl.fit_transform(x)
        y= väestö_[väestö_.Vuosi<alkuvuosi]['Kohorttimuutos']

        ridge.fit(X,y)

        x = nollat_[nollat_.Vuosi<alkuvuosi][nolla_selittäjät]
        X = scl2.fit_transform(x)
        y = nollat_[nollat_.Vuosi<alkuvuosi]['Muutos']

        svr.fit(X,y)

        v = väestö_.copy()
        v_20 = v[v.Vuosi==alkuvuosi]
        v = v[v.Vuosi<alkuvuosi]

        v_20.Kohorttimuutos =  ridge.predict(scl.transform(v_20[norm_selittäjät]))
        v_20.Ennusta = np.maximum(0,v_20.Lähtö + v_20.Kohorttimuutos)
        v = pd.concat([v,v_20],axis=0)

        n = nollat_.copy()

        n_20 = n[n.Vuosi==alkuvuosi]
        n = n[n.Vuosi<alkuvuosi]

        n_20.Muutos =  svr.predict(scl2.transform(n_20[nolla_selittäjät]))
        n_20.ennusta = np.maximum(0,n_20.Lähtö + n_20.Muutos)
        n = pd.concat([n,n_20],axis=0)

        if ennusteen_pituus > 1:
            for year in tqdm(range(alkuvuosi+1, alkuvuosi + 1 + ennusteen_pituus)):#last_year+1)):

                hed_df = v[(v.Vuosi == year -1) & (v.Ikä.isin(np.arange(hed_min,hed_max+1)))].groupby('Vuosi').agg({'Ennusta':'sum'}).rename(columns={'Ennusta':'Hed'}).reset_index().copy()


                nolla_df = n[(n.Vuosi==year-1)].copy()

                ykköset = nolla_df.copy()

                ykköset.Lähtö = ykköset.Ennusta
                ykköset.Vuosi+=1
                ykköset.Ikä+=1
                ykköset = ykköset[['Vuosi',

                                   'Ikä','Lähtö']]


                nolla_df.Lähtö = nolla_df.Ennusta
                nolla_df.Vuosi+=1


                nolla_df['Hed']=hed_df['Hed'].values


                loput = v[(v.Vuosi==year-1)&(v.Ikä<100)]
                loput.Ikä+=1
                loput.Vuosi+=1
                loput.Lähtö=loput.Ennusta
                loput.drop(['Ennusta','Kohorttimuutos'],axis=1, inplace=True)

                loput = pd.concat([ykköset,loput],axis=0)





                nolla_df['Muutos'] = svr.predict(scl2.transform(nolla_df[nolla_selittäjät]))
                nolla_df['Ennusta'] = np.maximum(0, nolla_df.Lähtö + nolla_df.Muutos)

                n = pd.concat([n,nolla_df], axis = 0)



                loput['Kohorttimuutos'] = ridge.predict(scl.transform(loput[norm_selittäjät]))
                loput['Ennusta'] = np.maximum(0,loput.Lähtö + loput.Kohorttimuutos)

                v = pd.concat([v,loput],axis = 0)


        result = pd.concat([n[['Vuosi',
                               'Ikä',
                               
                               'Ennusta']],v[['Vuosi',
                                              'Ikä',                                              
                                              'Ennusta']]],axis = 0)
        result = result.sort_values(by='Ikä')

        toteutunut = pd.concat([nollat_[(nollat_.Vuosi<alkuvuosi)][['Vuosi','Ikä','Ennusta']],väestö_[(väestö_.Vuosi<alkuvuosi)][['Vuosi','Ikä','Ennusta']]],axis=0)
        toteutunut = toteutunut.sort_values(by='Ikä')
        
        # Dokumentoi

        res_group = result.groupby('Vuosi').agg({'Ennusta':'sum'})
        res_group.Ennusta=np.ceil(res_group.Ennusta).astype(int)
        
        result['Kaupunki'] = city
        
        
        df = result.sort_values(by=['Vuosi','Ikä'])
        
        df['Ennuste/Toteutunut'] = df.apply(lambda x: apply_uncertainty(x['Vuosi'],alkuvuosi),axis=1)
        
        df.columns = [c.capitalize() for c in df.columns]
        df = df.set_index('Vuosi')
        
        df = df[['Kaupunki','Ikä', 'Ennuste/toteutunut','Ennusta']]
        df = df.rename(columns={'Ennusta':'Väestö','Ennuste/toteutunut':'Ennuste/Toteutunut'})
        
        meta_data = pd.DataFrame([{'Kaupunki':city,
                                   'Opetuksen aloitusvuosi':aloita,
                                   'Pienin hedelmällisyysikä':hed_min,
                                   'Suurin hedelmällisyysikä':hed_max,
                                   'Puiden lukumäärä':puut,
                                   'Simulaation MAE': nmae,
                                   'Simulaation RMSE': nrmse,
                                   'Simulaation R²':r2,
                                   'Simulaatiovuodet yhteensä': len(testattavat.split(',')),
                                   'Simulaatiovuodet': testattavat,
                                   'Testin MAE': quick_nmae,
                                   'Testin RMSE': quick_nrmse,
                                   'Testin R²':quick_r2,
                                   'Testatut vuodet':toteutuneet_ennusteet,
                                   'Viimeisin testivuosi': vika_vuosi,
                                   'Tilastokeskuksen MAE': nmae_tk,
                                   'Tilastokeskuksen RMSE': nrmse_tk,
                                   'Tilastokeskuksen R²':r2_tk,
                                   'Koko väestön toteuma vuodelle '+str(sorted(ennuste_leikkaus)[-1]): v_tot_document,
                                   'Ennuste': enn_document,
                                   'Tilastokeskuksen ennuste': v_enn_document,
                                   'Ennusteen ero toteumaan': diff_document,
                                   'Tilastokeskuksen ennusteen ero toteumaan':v_diff_document
                                  }]).T.reset_index().rename(columns={'index':'Ennusteen metatiedot',0:'Arvo'}).set_index('Ennusteen metatiedot')
       
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        df.to_excel(writer, sheet_name= 'Väestöennuste_'+city)#city+'_'+datetime.now().strftime('%d_%m_%Y'))
        tk_forecast.to_excel(writer, sheet_name = 'TK väestöennuste_'+city)
        simulation_test['Kaupunki'] = city
        simulation_test = simulation_test[['Kaupunki','Ikä','Ennusta']]
        simulation_test.reset_index().sort_values(by=['Vuosi','Ikä']).rename(columns={'vuosi':'Vuosi','ikä':'Ikä','Ennusta':'Väestöennuste'}).set_index('Vuosi').loc[testi_alkuvuosi:].to_excel(writer, sheet_name = 'Simulaatiodata')
        meta_data.to_excel(writer, sheet_name = 'Ennusteen metadata')
        writer.save()
        xlsx_io.seek(0)
        # https://en.wikipedia.org/wiki/Data_URI_scheme
        media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        data = base64.b64encode(xlsx_io.read()).decode("utf-8")
        href_data_downloadable = f'data:{media_type};base64,{data}'
        
        
        tk_plot = tk_forecast.reset_index().groupby('Vuosi').agg({'Tilastokeskuksen ennuste':'sum'}).sort_index()
        
        if ennusteen_pituus > 1:
            
            tk_plot = tk_plot.loc[:alkuvuosi+ennusteen_pituus]
            title = city.strip().capitalize()+': väestöennuste '+str(alkuvuosi)+' - '+str(result.Vuosi.max())
        else:
            
            tk_plot = tk_plot.loc[:alkuvuosi+ennusteen_pituus-1]
            title = city.strip().capitalize()+': väestöennuste '+str(alkuvuosi)
        
        tk_result = tk_forecast
        prediction_result = df
        
        tk_min = tk_plot.index.min()
        tk_max = tk_plot.index.max()
        
        tk_latest_age = tk_forecast.loc[vika_vuosi,:]
        
        tot_plot = res_group.loc[res_group.index <=alkuvuosi-1]
        enn_plot = res_group.loc[res_group.index.isin(range(alkuvuosi,alkuvuosi+11))]
        uncertain_plot = res_group.loc[res_group.index.isin(range(alkuvuosi+11,alkuvuosi+21))]
        very_uncertain_plot = res_group.loc[res_group.index.isin(range(alkuvuosi+21,res_group.index.max()+1))]
        
        test_plot = test_result.groupby('Vuosi').Ennusta.sum()
        test_tot_plot = test_toteutunut.groupby('Vuosi').Ennusta.sum()
        tk_test_plot = tk_plot.loc[tk_min:test_toteutunut.Vuosi.max()]

            

        return html.Div(children = [

            dcc.Graph(figure = go.Figure(
                data=[


                            go.Scatter(x = test_plot.index,
                                        y = np.ceil(test_plot.values),
                                         name = 'Ennuste',
                                       line = dict(color='red')
                                        ),



                            go.Scatter(x = test_tot_plot.index,
                                        y = test_tot_plot.values,
                                         name = 'Toteutunut',
                                         line = dict(color='green')
                                        ),
                            go.Scatter(x = tk_test_plot.index,
                                       y = tk_test_plot['Tilastokeskuksen ennuste'],
                                       name = 'Tilastokeskuksen ennuste',
                                       line = dict(color = 'blue')
                                      )
                            
                             ],
                       layout = go.Layout(xaxis = dict(title = 'Vuodet'),
                                          yaxis= dict(title = 'Väestö', 
                                                                  tickformat = ' '),
                                          title = dict(xref='paper', 
                                                       yref='paper', 
                                                       x=.5,

                                                       xanchor='left', 
                                                       yanchor='bottom',
                                                       text=city.strip().capitalize()+': väestöennustesimulaatio '+str(testi_alkuvuosi)+' - '+str(test_result.Vuosi.max()),
                                                       font=dict(family='Arial',
                                                                 size=30,
                                                                 color='black'
                                                                ),



                                                      )

                                          ))),



        html.Br(),
        html.Br(),
        html.H2('Simulaatioindikaattorit',style=dict(textAlign='center',fontSize=40, fontFamily='Arial')),
        html.Br(),
        html.Div(className = 'row', children =[
            html.Div(className = 'four columns', children = [
                html.H3('MAE (normalisoitu)',style=dict(textAlign='center')),
                daq.LEDDisplay(
                id='NMAE',
                value = nmae,
                size =120,
                color = nmae_color,
                backgroundColor='black'
                    
                ),
                html.H3(nmae_text,style=dict(textAlign='center', color = nmae_color))
            ]),
            html.Div(className = 'four columns', children = [
                html.H3('RMSE (normalisoitu)',style=dict(textAlign='center')),
                daq.LEDDisplay(
                id='NRMSE',
                value = nrmse,
                size =120,
                color = nrmse_color,
                backgroundColor='black'
                    
                    
                ),
                html.H3(nrmse_text,style=dict(textAlign='center',color = nrmse_color))
            ]),
            html.Div(className = 'four columns', children = [
                html.H3('R²',style=dict(textAlign='center')),
                daq.LEDDisplay(
                id='R2',
                value = r2,
                size =120,
                color = r2_color,
                backgroundColor='black'
                    
                ),
                html.H3(r2_text,style=dict(textAlign='center', color = r2_color))
            ])
        ]
                ),
        html.Br(),
        html.P(quick_chain, style = dict(textAlign='center', color = 'purple', fontWeight='bold', fontFamily='Arial',fontSize=16)),
        html.P(tk_chain, style = dict(textAlign='center', color = 'blue', fontWeight='bold', fontFamily='Arial',fontSize=16)),
        #html.P('Alla koneoppimismallin ja Tilastokeskuksen ennusteen vertailua iän mukaan vuodelle '+str(vika_vuosi)+'.', style = dict(textAlign='center', color = 'black', fontWeight='bold', fontFamily='Arial',fontSize=16)),
        html.Br(),
#         dcc.Graph(figure = go.Figure(
#                             data =[
#                                 go.Scatter(
#                                    x = age_test.ikä,
#                                    y = np.ceil(age_test.ennusta),
#                                    name = 'Ennuste',
#                                     mode='markers',
#                                    line = dict(color = 'red')
#                                 ),
#                                 go.Scatter(
#                                    x = age_true.ikä,
#                                    y = age_true.ennusta,
#                                     mode='markers',
#                                    name = 'Toteutunut',
#                                    line = dict(color = 'green')
#                                 ),
#                                 go.Scatter(
#                                    x = tk_latest_age.Ikä,
#                                    y = tk_latest_age['Tilastokeskuksen ennuste'],
#                                    name = 'Tilastokeskuksen ennuste',
#                                     mode='markers',
#                                    line = dict(color = 'blue')
#                                 )
#                             ],
#                     layout= go.Layout(
#                                     xaxis = dict(title = 'Ikä'),
#                                     yaxis= dict(title = 'Väestö',
#                                                 tickformat = ' '
#                                                ),
#                                     title = dict(xref='paper', 
#                                                  yref='paper', 
#                                                  xanchor='left', 
#                                                  yanchor='bottom',
#                                                  text=city.strip().capitalize()+': väestöennustetesti iän mukaan vuodelle '+str(vika_vuosi),
#                                                  font=dict(family='Arial',
#                                                                  size=30,
#                                                                  color='black'
#                                                                 )
#                                  )
#                  )
#         )
#                  ),
        dcc.Graph(figure = go.Figure(
            data=[

                            go.Scatter(x = tot_plot.index,
                                       y = tot_plot.Ennusta,
                                       name = 'Toteutunut',
                                       line = dict(color='green')
                                      ),
                            go.Scatter(x = enn_plot.index,
                                       y = enn_plot.Ennusta,
                                       name = 'Ennuste',
                                       line = dict(color='orange')

                                        ),
                            go.Scatter(x = uncertain_plot.index,
                                       y = uncertain_plot.Ennusta,
                                       name = 'Epävarma ennuste',
                                       line = dict(color='purple')

                                        ),
                            go.Scatter(x = very_uncertain_plot.index,
                                       y = very_uncertain_plot.Ennusta,
                                       name = 'Erittäin epävarma ennuste',
                                       line = dict(color='red')

                                        ),
                            go.Scatter(x = tk_plot.index, 
                                       y = tk_plot['Tilastokeskuksen ennuste'],
                                      name = 'Tilastokeskuksen ennuste '+str(tk_min)+' - '+str(tk_max),
                                      line = dict(color = 'blue')
                                      )
                             ],
                       layout = go.Layout(xaxis = dict(title = 'Vuodet'),
                                          yaxis= dict(title = 'Väestö',
                                                      tickformat = ' '
                                                     ),
                                          title = dict(xref='paper', 
                                                       yref='paper', 
                                                       x=.5,

                                                       xanchor='left', 
                                                       yanchor='bottom',
                                                       text=title,
                                                       font=dict(family='Arial',
                                                                 size=30,
                                                                 color='black'
                                                                )



                                                      )

                                          )
                       )
             ),
            


           
           
#            html.H2(city+':', style= dict(textAlign='center', color = 'black', fontWeight='bold', fontFamily='Arial',fontSize=30)),
#            html.Div(id = 'year_selection_indicator',
#                      style= dict(textAlign='center', color = 'black', fontWeight='bold', fontFamily='Arial',fontSize=30)
#                     ),
#             html.Div(id='ikägraafi'),
            

#           html.H2('Valitse ennustevuosi, jota tarkastella iän mukaan.'),
#           dcc.Dropdown(id ='vuosivalitsin',
#                      multi=False,
#                      options = [{'label':s, 'value': s} for s in range(alkuvuosi,alkuvuosi+ennusteen_pituus+1)],
#                      value=alkuvuosi
#                      ),

            html.Br(),
                    html.A(
                        'Lataa yksityiskohtainen Excel-taulukko. ',
                        id='excel-download',
                        download=city+'_'+datetime.now().strftime('%d_%m_%Y_%H:%M')+".xlsx",
                        href=href_data_downloadable,
                        target="_blank"
                    ),
            html.Br(),
            html.Br()
        ])




    
# @app.callback(
#     Output('ikägraafi','children'),
    
#     [

#    Input('vuosivalitsin', 'value')
#    #Input('kunnat','value')
#     ]
# )
# def update_age_graph(year):
    
    
    

    
#     try:
            
#         tk_plot = tk_result.loc[year,:]
#         ennuste_plot = prediction_result.loc[year,:]
#         return dcc.Graph(figure = go.Figure(data=[
#                                     go.Scatter(x = ennuste_plot.Ikä,
#                                               y = np.ceil(ennuste_plot.Väestö),
#                                               name = 'Ennuste',
#                                                mode = 'markers',
#                                               line = dict(color='red')),
#                                     go.Scatter(x = tk_plot.Ikä,
#                                               y = tk_plot['Tilastokeskuksen ennuste'],
#                                               name = 'Tilastokeskuksen ennuste',
#                                                mode = 'markers',
#                                               line = dict(color='blue'))],
#                                     layout = go.Layout(showlegend=True, xaxis = dict(title = 'Ikä'),
#                                               yaxis= dict(title = 'Väestö', 
#                                                                       tickformat = ' '),
#                                               title = dict(xref='paper', 
#                                                            yref='paper', 

#                                                            xanchor='left', 
#                                                            yanchor='bottom',
#                                                            text=' 0 - 100 -vuotiaat',
#                                                            font=dict(family='Arial',
#                                                                      size=30,
#                                                                      color='black'
#                                                                     )



#                                                           )

#                                               )
#                            )
#                                            )
#     except:
#         ennuste_plot = prediction_result.loc[year,:]
           
#         return dcc.Graph(figure = go.Figure(data=[
#                                     go.Scatter(x = ennuste_plot.Ikä,
#                                               y = np.ceil(ennuste_plot.Väestö),
#                                               name = 'Ennuste',
#                                               mode = 'markers',
#                                               line = dict(color='red'))

#                                             ],
#                                     layout = go.Layout(showlegend=True,
#                                                        xaxis = dict(title = 'Ikä'),

#                                               yaxis= dict(title = 'Väestö', 
#                                                                       tickformat = ' '),
#                                               title = dict(xref='paper', 
#                                                            yref='paper', 

#                                                            xanchor='left', 
#                                                            yanchor='bottom',
#                                                            text=' 0 - 100 -vuotiaat',
#                                                            font=dict(family='Arial',
#                                                                      size=30,
#                                                                      color='black'
#                                                                     )



#                                                           )

#                                               )
#                            )
#                                            )

    
                    

    
app.layout= serve_layout
#Aja sovellus.
if __name__ == '__main__':
    app.run_server(debug=False)

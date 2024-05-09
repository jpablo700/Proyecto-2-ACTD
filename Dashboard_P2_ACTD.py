# -*- coding: utf-8 -*-

# Ejecute esta aplicación 
# y luego visite el sitio
# http://127.0.0.1:8050/ 
# en su navegador.

import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output, State
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import psycopg2
import plotly.graph_objects as go
from dotenv import load_dotenv # pip install python-dotenv
import os


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# path to env file
ruta_actual=os.path.dirname(os.path.abspath(__file__))
env_path=os.path.join(ruta_actual,'app.env')
#env_path="C:\\Users\\20192818\\OneDrive - TU Eindhoven\\Documents\\Uniandes Intercambio\\Analitica Computacional\\Proyecto 2\\env\\app.env"
# load env 
load_dotenv(dotenv_path=env_path)
# extract env variables
USER=os.getenv('USER')
PASSWORD=os.getenv('PASSWORD')
HOST=os.getenv('HOST')
PORT=os.getenv('PORT')
DBNAME=os.getenv('DBNAME')

#connect to DB
print(DBNAME)
print(USER)
print(PASSWORD)
print(HOST)
print(PORT)
engine = psycopg2.connect(
    dbname=DBNAME,
    user=USER,
    password=PASSWORD,
    host=HOST,
    port=PORT
)

cursor = engine.cursor()

query1 = """
SELECT SEX, COUNT(*) AS total_clients,
       SUM(is_default) AS defaults
FROM datos
GROUP BY SEX;
"""
cursor.execute(query1)
result1 = cursor.fetchall()

proportion_default1 = {}
for row in result1:
    sex = row[0]
    total_clients = row[1]
    defaults = row[2]
    proportion_default1[sex] = defaults / total_clients

x_labels = ['Male' if key == 1 else 'Female' for key in proportion_default1.keys()]

fig1 = go.Figure(data=go.Bar(x=x_labels, y=list(proportion_default1.values()), marker_color=['blue', 'pink']))

fig1.update_layout(title='Percentage of Defaults by Gender',
                   xaxis=dict(title='Gender'),
                   yaxis=dict(title='Default percentage'))

query2 = """
SELECT AGE, SEX, is_default
FROM datos;
"""
cursor.execute(query2)
result2 = cursor.fetchall()
columns2 = ['AGE', 'SEX', 'is_default']
data2 = pd.DataFrame(result2, columns=columns2)
defaulters2 = data2[data2['is_default'] == 1]


# Bin AGE into age groups
age_bins = range(0, defaulters2['AGE'].max() + 10, 10)
age_labels = [f"{i}-{i+10}" for i in age_bins[:-1]]
defaulters2['age_group'] = pd.cut(defaulters2['AGE'], bins=age_bins, labels=age_labels, right=False)

distribucion_edad2 = defaulters2.groupby(['age_group', 'SEX']).size().unstack()

male2 = distribucion_edad2[1]
female2 = distribucion_edad2[2]
y2 = list(range(0, len(distribucion_edad2)))
age_group_labels = list(distribucion_edad2.index)

fig2 = go.Figure()

fig2.add_trace(go.Bar(y=y2, x=male2, orientation='h', name='Males', marker_color='royalblue'))
fig2.add_trace(go.Bar(y=y2, x=female2, orientation='h', name='Females', marker_color='lightpink'))

fig2.update_layout(title="Distribution of people that defaulted per age group",
                  xaxis=dict(title='Count'),
                  yaxis=dict(title='Age Group', tickvals=y2, ticktext=age_group_labels))

query3 = """
SELECT
    SEX,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) AS total_defaults
FROM
    datos
GROUP BY
    SEX;
"""
cursor.execute(query3)
result3 = cursor.fetchall()

# Initialize defaults for males and females
males_defaults = 0
females_defaults = 0

# Update defaults based on the retrieved data
for row in result3:
    sex = row[0]
    defaults = row[1]
    if sex == 1:  # Male
        males_defaults = defaults
    elif sex == 2:  # Female
        females_defaults = defaults

fig3 = go.Figure(data=[go.Bar(x=['Male', 'Female'], y=[males_defaults, females_defaults], marker_color=['blue', 'pink'])])

fig3.update_layout(title='Total Defaults by Gender',
                   xaxis=dict(title='Gender'),
                   yaxis=dict(title='Total Defaults'))

# This line does the deserialization
model=keras.models.load_model(os.path.join(ruta_actual, 'default_pred.keras'))
#model = keras.models.load_model('C:\\Users\\20192818\\OneDrive - TU Eindhoven\\Documents\\Uniandes Intercambio\\Analitica Computacional\\Proyecto 2\\models\\default_pred.keras')


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Img(src='https://img.freepik.com/free-vector/wallet-concept-illustration_114360-4069.jpg?t=st=1714747003~exp=1714750603~hmac=c6ee94ca4e31963fa9375c64407d4aa01d8141bba8e6df0d8c8f1bb4a928cdbe&w=740', style={'width': '100px', 'height': '100px', 'margin-right': '10px'}),
            html.H1(children='Default of Credit Card Clients',style={'background-color': '#003f5c', 'color':'white', 'font-weight':'bold'}),
        ], style={'display': 'flex', 'align-items': 'center'}),
    ],style={'background-color': '#003f5c', 'height': '100px','margin-top': '10px'}),

    html.H5(children='Understanding the probability of default allows credit card companies to assess the risk associated with extending credit to individual clients. This helps them determine whether to approve a credit card application. This application calculates the probability of default of (potential) credit card clients. It also shows some visualizations of the risk of certain client segments (based on gender and age).'),
    
    html.Div([
        html.Div([
            html.H3(children='Probability of default payment class',style={'background-color': 'white', 'color':'#003f5c', 'font-weight':'bold'}),
        ], style={'display': 'flex', 'align-items': 'center'}),
    ],style={'background-color': 'white', 'height': '50px','margin-top': '20px', 'margin-bottom':'20px'}),
    
    
    html.H5(children='Enter the values of the following characteristics to make the default class probability prediction.'),

       html.Div(["Gender: ",
        dcc.Dropdown(id='SEX', value='Male', options=['Male', 'Female'])]), 
    html.Br(),
    html.Div(["Education: ",
              dcc.Dropdown(id='EDUCATION', value='graduate school', options=['graduate school', 'university', 'high school', 'others'])]), 
    html.Br(),
    html.Div(["Marital status: ",
              dcc.Dropdown(id='MARRIAGE', value='married', options=['married', 'single', 'others'])]), 
    html.Br(),  
    html.Div(["Age: "]),
    html.Div([dcc.Input(id='AGE', type='text', value='')]), 
    html.Br(),
    html.Div(["Amount of the given credit (NT dollar): "]),
    html.Div([dcc.Input(id='LIMIT_BAL', type='text', value='')]), 
    html.Br(),
    html.Div(["Repayment status in September, 2005: "]),
    html.Div([dcc.Input(id='PAY_0', type='text', value='')]), 
    html.Br(),
    html.Div(["Repayment status in August, 2005: "]),
    html.Div([dcc.Input(id='PAY_2', type='text', value='')]), 
    html.Br(),
    html.Div(["Repayment status in July, 2005: "]),
    html.Div([dcc.Input(id='PAY_3', type='text', value='')]), 
    html.Br(),
    html.Div(["Repayment status in June, 2005: "]),
    html.Div([dcc.Input(id='PAY_4', type='text', value='')]), 
    html.Br(),
    html.Div(["Repayment status in May, 2005: "]),
    html.Div([ dcc.Input(id='PAY_5', type='text', value='')]), 
    html.Br(),
    html.Div(["Repayment status in April, 2005: "]),
    html.Div([dcc.Input(id='PAY_6', type='text', value='')]), 
    html.Br(),
    html.Div(["Amount of bill statement in September, 2005: "]),
    html.Div([dcc.Input(id='BILL_AMT1', type='text', value='')]), 
    html.Br(),
    html.Div(["Amount of bill statement in August, 2005: "]),
    html.Div([dcc.Input(id='BILL_AMT2', type='text', value='')]), 
    html.Br(),
    html.Div(["Amount of bill statement in July, 2005: "]),
    html.Div([dcc.Input(id='BILL_AMT3', type='text', value='')]), 
    html.Br(),
    html.Div(["Amount of bill statement in June, 2005: "]),
    html.Div([dcc.Input(id='BILL_AMT4', type='text', value='')]), 
    html.Br(),
    html.Div(["Amount of bill statement in May, 2005: "]),
    html.Div([dcc.Input(id='BILL_AMT5', type='text', value='')]), 
    html.Br(),
    html.Div(["Amount of bill statement in April, 2005: "]),
    html.Div([dcc.Input(id='BILL_AMT6', type='text', value='')]), 
    html.Br(),
    html.Div(["Amount paid in September, 2005: "]),
    html.Div([dcc.Input(id='PAY_AMT1', type='text', value='')]), 
    html.Br(),
    html.Div(["Amount paid in August, 2005: "]),
    html.Div([dcc.Input(id='PAY_AMT2', type='text', value='')]), 
    html.Br(),
    html.Div(["Amount paid in July, 2005: "]),
    html.Div([dcc.Input(id='PAY_AMT3', type='text', value='')]), 
    html.Br(),
    html.Div(["Amount paid in June, 2005: "]),
    html.Div([dcc.Input(id='PAY_AMT4', type='text', value='')]), 
    html.Br(),
    html.Div(["Amount paid in May, 2005: "]),
    html.Div([dcc.Input(id='PAY_AMT5', type='text', value='')]), 
    html.Br(),
    html.Div(["Amount paid in April, 2005: "]),
    html.Div([dcc.Input(id='PAY_AMT6', type='text', value='')]), 
    html.Br(),
    html.Div(["Probability of default payment class: ", html.Div(id='output-prob', style={'fontSize': '48px'})]),
    html.Br(),


    html.Div([
        html.Div([
            html.H3(children='Comparing males vs females total defaults',style={'background-color': 'white', 'color':'#003f5c', 'font-weight':'bold'}),
        ], style={'display': 'flex', 'align-items': 'center'}),
    ],style={'background-color': 'white', 'height': '40px','margin-top': '20px', 'margin-bottom':'20px'}),
   
    
    html.H5("The following graph shows the total amount of defaults per gender."),
    dcc.Graph(figure=fig3),

    html.Div([
        html.Div([
            html.H3(children='Comparing males vs females precentage of defaults',style={'background-color': 'white', 'color':'#003f5c', 'font-weight':'bold'}),
        ], style={'display': 'flex', 'align-items': 'center'}),
    ],style={'background-color': 'white', 'height': '40px','margin-top': '20px', 'margin-bottom':'20px'}),

    html.Div([
        html.H5("The following graph shows the percentage of people per gender that defaulted."),
        dcc.Graph(figure=fig1)
        ]),
 
     html.Div([
        html.Div([
            html.H3(children='Comparing males vs females defaults including age groups',style={'background-color': 'white', 'color':'#003f5c', 'font-weight':'bold'}),
        ], style={'display': 'flex', 'align-items': 'center'}),
    ],style={'background-color': 'white', 'height': '40px','margin-top': '20px', 'margin-bottom':'20px'}),

    html.H5("The following graph shows the amount of people that defaulted seperated by gender and age groups."),
    dcc.Graph(figure=fig2),

]
)    

@app.callback(
    Output(component_id='output-prob', component_property='children'),
    [
        Input(component_id='LIMIT_BAL', component_property='value'),
        Input(component_id='SEX', component_property='value'),
        Input(component_id='EDUCATION', component_property='value'),
        Input(component_id='MARRIAGE', component_property='value'),
        Input(component_id='AGE', component_property='value'),
        Input(component_id='PAY_0', component_property='value'),
        Input(component_id='PAY_2', component_property='value'),
        Input(component_id='PAY_3', component_property='value'),
        Input(component_id='PAY_4', component_property='value'),
        Input(component_id='PAY_5', component_property='value'),
        Input(component_id='PAY_6', component_property='value'),
        Input(component_id='BILL_AMT1', component_property='value'),
        Input(component_id='BILL_AMT2', component_property='value'),
        Input(component_id='BILL_AMT3', component_property='value'),
        Input(component_id='BILL_AMT4', component_property='value'),
        Input(component_id='BILL_AMT5', component_property='value'),
        Input(component_id='BILL_AMT6', component_property='value'),
        Input(component_id='PAY_AMT1', component_property='value'),
        Input(component_id='PAY_AMT2', component_property='value'),
        Input(component_id='PAY_AMT3', component_property='value'),
        Input(component_id='PAY_AMT4', component_property='value'),
        Input(component_id='PAY_AMT5', component_property='value'),
        Input(component_id='PAY_AMT6', component_property='value')
    ]
)
def update_output_div(limit_bal, sex, education, marriage, age, pay_0, pay_2, pay_3, pay_4, pay_5, pay_6, bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6, pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6):
    # Convert values from dropdown options to database values
    sex_mapping = {'Male': 1, 'Female': 2}
    education_mapping = {'graduate school': 1, 'university': 2, 'high school': 3, 'others': 4}
    marriage_mapping = {'married': 1, 'single': 2, 'others': 3}

    sex_value = sex_mapping.get(sex)
    education_value = education_mapping.get(education)
    marriage_value = marriage_mapping.get(marriage)

    if limit_bal == '' or age == '' or any(pay == '' for pay in [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6]) or any(bill_amt == '' for bill_amt in [bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6]) or any(pay_amt == '' for pay_amt in [pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]):
        return None  # Return None if any input variable is empty

    # Manually scale each variable using mean and std
    scaled_limit_bal = (float(limit_bal) - (-0.057498112)) / 0.970437468
    scaled_age = (float(age) - 0.008070581) / 1.012224999
    scaled_pay_0 = (float(pay_0) - 0.115899433) / 1.089776934
    scaled_pay_2 = (float(pay_2) - 0.105950963) / 1.065957247
    scaled_pay_3 = (float(pay_3) - 0.098341613) / 1.073251211
    scaled_pay_4 = (float(pay_4) - 0.090143557) / 1.073637603
    scaled_pay_5 = (float(pay_5) - 0.086636032) / 1.081438062
    scaled_pay_6 = (float(pay_6) - 0.091598083) / 1.085813542
    scaled_bill_amt1 = (float(bill_amt1) - (-0.00284588)) / 0.999135844
    scaled_bill_amt2 = (float(bill_amt2) - (-0.002395307)) / 0.997570881
    scaled_bill_amt3 = (float(bill_amt3) - (-0.00377848)) / 0.992530649
    scaled_bill_amt4 = (float(bill_amt4) - 0.001327004) / 1.000719493
    scaled_bill_amt5 = (float(bill_amt5) - 0.006057525) / 1.007324081
    scaled_bill_amt6 = (float(bill_amt6) - 0.005765229) / 1.003909369
    scaled_pay_amt1 = (float(pay_amt1) - (-0.029184767)) / 0.873674289
    scaled_pay_amt2 = (float(pay_amt2) - (-0.028994223)) / 0.711163368
    scaled_pay_amt3 = (float(pay_amt3) - (-0.018167954)) / 0.856422065
    scaled_pay_amt4 = (float(pay_amt4) - (-0.014411757)) / 0.920212386
    scaled_pay_amt5 = (float(pay_amt5) - (-0.017049373)) / 0.983784625
    scaled_pay_amt6 = (float(pay_amt6) - (-0.004321353)) / 1.053369428

    # Construct the input x
    x =np.array( [[scaled_limit_bal, sex_value, education_value, marriage_value, scaled_age,
        scaled_pay_0, scaled_pay_2, scaled_pay_3, scaled_pay_4, scaled_pay_5, scaled_pay_6,
        scaled_bill_amt1, scaled_bill_amt2, scaled_bill_amt3, scaled_bill_amt4, scaled_bill_amt5, scaled_bill_amt6,
        scaled_pay_amt1, scaled_pay_amt2, scaled_pay_amt3, scaled_pay_amt4, scaled_pay_amt5, scaled_pay_amt6
    ]])
    
    # Check inputs are correct 
    ypred = model.predict(x)

    # Convert ypred to float and format the probability as a percentage
    prob_percentage = '{:.2f}%'.format(float(ypred[0][1]) * 100)
    
    return prob_percentage




if __name__ == '__main__':
    app.run_server(debug=True)


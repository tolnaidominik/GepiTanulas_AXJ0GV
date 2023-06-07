import pickle
import numpy as np
import pandas as pd
import datetime
import streamlit as st
import plotly.graph_objects as go

st.write("""
# Chocolate bar rating prediction
## This app is going to predict your chocolate bar rating for sure :).
""")

link = '[Kaggle - Chocolate Bar Ratings](https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings)'
st.markdown(link, unsafe_allow_html=True)

st.sidebar.header('Input Parameters')

def user_input_features():
    ReviewDate = st.sidebar.slider('Review date', 1, 1800, 2023)
    REF = st.sidebar.slider('REF', 1, 1, 2000)
    CocoaPercent = st.sidebar.slider('Cocoa Percent', 1, 1, 100)
    
    data = {'ReviewDate': ReviewDate,
            'REF': REF,
            'CocoaPercent': CocoaPercent
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Show user inputs
st.subheader('Input parameters')
st.write(df)

# Create Plotly plot
columns = ['ReviewDate', 'REF', 'CocoaPercent']

# create a new DataFrame with the selected columns
df_game = df[columns]
# Convert the first row of the DataFrame to a list
y = df_game.values.tolist()[0]

fig = go.Figure(data=go.Bar(x=columns, y=y), layout_title_text='Cholocate bar information')
st.plotly_chart(fig, use_container_width=True)

model_final_pipe = pickle.load(open('model_trained.pkl', 'rb'))

prediction = model_final_pipe.predict(df)

st.subheader('Predicted Chocolate bar rating')
prediction = int(np.round(prediction, 0))
st.title(prediction)

import streamlit as st
import os
import numpy as np
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
 

df = pickle.load(open('movies_data.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

warnings.filterwarnings("ignore")

@st.cache

 
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


def engine(movie_name):

    try:

        index=int(get_index_from_title(movie_name))
        # print(index,movie_name)
        movie_rec_scores = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
        movie_rec=""
        for i in movie_rec_scores[1:6]:
            movie_rec += " [ "+str(df.iloc[i[0]].title) +"] "
        return movie_rec

    except Exception as e:
        return "[sorry unable to fetch recommendations ]"
    
 
  

# print(make_prediction("Would not go back"))

st.title("Movie Recommender  App")
st.write(
    "A simple machine learning app "
)

form = st.form(key="my_form")
movie_name= form.text_input(label="Enter the name of movie")
submit = form.form_submit_button(label="get similar movies")

if submit:
    # make prediction from the input text
    result=engine(movie_name)
 
    # Display results of the NLP task
    st.header("similar to your movie :")
 
  
    st.write( result)
    
        
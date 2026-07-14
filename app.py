import streamlit as st
import pandas as pd
import pickle as pk
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
words = stopwords.words("english")
stemmer = PorterStemmer()

st.title("Text Classification")
text = st.text_area("Enter text to Classify:") 

if not text:
    st.warning("Please enter some text to classify!!!")


with open("LogisticRegression.pickle","rb") as f:
    model = pk.load(f)

if st.button("Perdict") and text:
    data = pd.DataFrame({"text":[text]})
    st.dataframe(data)

    data['text'] = list(map(lambda x: " ".join([i for i in x.lower().split() if i not in words]), data['text']))  
    data['cleaned'] = data['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() ]).lower())

    predication = model.predict(data['cleaned'])
    st.success(f"Predication :  {predication[0]}")
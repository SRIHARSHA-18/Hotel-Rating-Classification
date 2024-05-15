# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:12:19 2023

@author: afrin
"""

import streamlit as st
import pickle
#import numpy as np
import time
import re #regular expression
import string
#import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

model = pickle.load(open('C:/Users/User/Downloads/svm_model.pkl','rb'))

st.markdown(
    """
    <style>
    .main {
        background-color:  #008080;
    }
    </style>
    """,
    unsafe_allow_html=True
)

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

def preprocessing(text):
    #'''Make text lowercase, remove text in square brackets, remove punctuation & remove words containing numbers,remove urls.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9" "]+"," ",text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '',text)
    text = re.sub("[^A-Za-z" "]+"," ", text)
    
    #'''Tokenization'''
    tokens = word_tokenize(text)
    
    #'''Removing Stop words'''
    
    tokens = [token for token in tokens if token not in stop_words]
    
    #'''Stemming'''
    #ps = PorterStemmer()
    #tokens = [ps.stem(word) for word in tokens]
    
    #'''Lemmatization'''
    
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove extra whitespaces and join tokens back to form cleaned text
    clean_Review = ' '.join(lemmatized_tokens).strip()
    
    return clean_Review


# Read positive and negative keywords from external text files
def read_keywords(filename):
    with open(filename, 'r') as file:
        keywords = [line.strip() for line in file]
    return keywords

positive_keywords = read_keywords('C:/Users/User/Downloads/positive.txt')
negative_keywords = read_keywords('C:/Users/User/Downloads/negative.txt')

import emoji

positive_emoji = "\U0001F604" 
negative_emoji = "\U0001F61E"

# Increase the size of the positive emoji
enlarged_positive_emoji = f"{positive_emoji * 4}"

# Increase the size of the negative emoji
enlarged_negative_emoji = f"{negative_emoji * 4}"




st.title("Hotel Review Sentiment Analysis")

# Create a text input box for user input
user_input = st.text_area("Enter a hotel review:")

user_input = preprocessing(user_input)

if st.button('Predict'):
    start = time.time()
    prediction = model.predict([user_input])
    end = time.time()
    print(prediction)
    st.write('Prediction time taken:',round(end - start,2),'seconds')
    st.write('Predicted Sentiment is:',prediction[0])
    
    
# Check for positive or negative keywords
    keywords_found = []
    if prediction == "Positive":
        enlarged_positive_emoji
        for keyword in positive_keywords:
            if keyword in user_input.lower():
                keywords_found.append(keyword)
              
    elif prediction == "Negative":
        enlarged_negative_emoji
        for keyword in negative_keywords:
            if keyword in user_input.lower():
                keywords_found.append(keyword)
                
                

    if keywords_found:
        st.write(f"Keywords associated with {prediction} prediction: {', '.join(keywords_found)}")
    else:
        st.write(f"No specific keywords associated with {prediction} prediction found in the review.")
st.balloons()

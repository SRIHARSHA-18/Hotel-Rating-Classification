# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:46:51 2023

@author: Srinivasa
"""

import streamlit as st

st.title("HOTEL REVIEWS ANALYSIS")
review = st.text_area("Enter a review:")

if st.button("Analyze"):
    # Perform sentiment analysis and emotion mining here
    st.write("Sentiment: Positive")
    st.write("Emotion: Happy")
    st.write("Sentiment: Negative")
    st.write("Emotion: Not Happy")

# Add more Streamlit elements for displaying visualizations

# Run the Streamlit app
if __name__ == "__main__":
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
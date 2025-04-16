import streamlit as st
from preprocessing import load_and_clean_data
from feature_based_recommender import recommender_by_features
# from lyrics_based_recommender import recomender_by_lyrics
from utils import get_all_songs_array
from sklearn.feature_extraction.text import CountVectorizer


st.title('Music Recommendation System')
st.markdown('<p style="font-size:24px; font-style: italic;">Find new music to love</p>', unsafe_allow_html=True)

# Load data
tracks = load_and_clean_data("tcc_ceds_music.csv")

# Input selection
selection = st.selectbox('Choose a favourite song of yours:', get_all_songs_array(tracks))
song = selection.split('  -  ')[0].lower()

if st.button('Show me recommendations!'):
    vectorizer = CountVectorizer()
    vectorizer.fit(tracks['string_data'])
    recommendations = recommender_by_features(song, tracks, vectorizer, 3)

    st.write('Here are some songs you might enjoy:')
    for rec in recommendations.values:
        st.markdown(f'#### *{rec[0]}* by {rec[1]}')

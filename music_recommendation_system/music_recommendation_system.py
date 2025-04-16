from decimal import Decimal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


tracks = pd.read_csv("tcc_ceds_music.csv")
print(tracks.info())
print(tracks.head())

#drop unneeded columns
tracks = tracks.drop(['Unnamed: 0',
                      'len',
                      ], axis = 1)

#drop rows with null values
tracks.dropna(inplace = True)

#drop rows with duplicated track names
tracks.drop_duplicates(subset=['track_name'], keep='first', inplace=True)

#all text columns lower case
tracks['artist_name'].apply(lambda x: x.lower())
tracks['track_name'].apply(lambda x: x.lower())
tracks['genre'].apply(lambda x: x.lower())
tracks['lyrics'].apply(lambda x: x.lower())
tracks['topic'].apply(lambda x: x.lower())

#add release_decade column
tracks['release_decade'] = (tracks['release_date']//10)*10

#drop column release_date
tracks = tracks.drop(['release_date'], axis = 1)

#returns string with no whitespaces
def no_whitespace(text):
    res = str(text)
    return res.replace(' ', '')

#add string_data column consisting of artist_name, genre and topic
tracks['string_data'] = (tracks['artist_name'].apply(no_whitespace) + ' ' 
                         + tracks['genre'].apply(no_whitespace)  + ' ' 
                         + tracks['topic'].apply(no_whitespace))

#subset of the main dataset without the musical features
tracks_topic = tracks.drop(['danceability', 
                            'loudness', 
                            'acousticness', 
                            'instrumentalness',
                            'valence',
                            'energy'], axis = 1)

#subset of the main dataset without the topic subfeatures
tracks_music = tracks.drop(['dating', 
                            'violence',
                            'world/life', 
                            'night/time',
                            'shake the audience',
                            'family/gospel',
                            'romantic', 
                            'communication', 
                            'obscene', 
                            'music',
                            'movement/places',
                            'light/visual perceptions',
                            'family/spiritual', 
                            'like/girls', 
                            'sadness', 
                            'feelings',], axis = 1)

track_vectorizer = CountVectorizer() 
vectorized = track_vectorizer.fit(tracks['string_data'])

#returns an array with similarities by features for every row
def get_features_similarities(song_name, data):
    #vector and numeric features for input song
    text_array1 = track_vectorizer.transform(data[data['track_name']==song_name]['string_data']).toarray()
    num_array1 = data[data['track_name']==song_name].select_dtypes(include=np.number).to_numpy()
   
    #storing similarity for each row
    sim = []
    for idx, row in data.iterrows():
        name = row['track_name']
        #vector and numeric features for current song
        text_array2 = track_vectorizer.transform(data[data['track_name']==name]['string_data']).toarray()
        num_array2 = data[data['track_name']==name].select_dtypes(include=np.number).to_numpy()
 
        #calculating similarities for text and numeric features
        text_sim = Decimal(cosine_similarity(text_array1, text_array2)[0][0])
        num_sim = Decimal(cosine_similarity(num_array1, num_array2)[0][0])
        sim.append(text_sim + num_sim)
     
    return sim

#returns song name and artist name of the songs most similar to the input song
def recommender_by_features(song_name, data=tracks, number_recommended=5):
    if number_recommended <= 0:
        return
    #adding a column similarity_factor and 
    # setting its values the values of the array returned by get_features_similarities
    data['similarity_factor'] = get_features_similarities(song_name, data)
    
    #sorting the datset by similarity in a descending order
    data.sort_values(by=['similarity_factor'],
                   ascending = [False],
                   inplace=True)
    
    return(data[['track_name', 'artist_name']][1:(number_recommended + 1)])


#lyrics tokenization
stemmer = PorterStemmer()

def token(lyrics):
    token = nltk.word_tokenize(lyrics)
    stemmed = []
    for w in token:
        stemmed.append(stemmer.stem(w))

    return " ".join(stemmed)

tracks['lyrics'].apply(lambda x: token(x))

tfid = TfidfVectorizer(analyzer='word', stop_words='english')
tfidVectorized = tfid.fit(tracks['lyrics'])

def recomender_by_lyrics(song_name, data=tracks, number_recommended=5):
    if number_recommended <= 0:
        return
    #vector and numeric features for input song
    text_array1 = tfid.transform(data[data['track_name']==song_name]['lyrics']).toarray()
   
    #storing similarity for each row
    sim = []
    for idx, row in data.iterrows():
        name = row['track_name']
        #vector and numeric features for current song
        text_array2 = tfid.transform(data[data['track_name']==name]['lyrics']).toarray()
 
        #calculating similarities
        text_sim = Decimal(cosine_similarity(text_array1, text_array2)[0][0])
        sim.append(text_sim)
    
    #adding a column lyrics_similarity and setting its values the values of the array sim
    data['lyrics_similarity'] = sim

    #sorting the datset by lyrics similarity in a descending order
    data.sort_values(by=['lyrics_similarity'],
                   ascending = [False],
                   inplace=True)
    
    return(data[['track_name', 'artist_name']][1:(number_recommended + 1)])

def recommend_songs(song_name, data=tracks, number_recommended_by_features = 3, number_recommended_by_lyrics = 2):
    #Entered track is not in the dataset
    if tracks[tracks['track_name'] == song_name].shape[0] == 0:
        print('Wrong track name may have been entered.')
        return
    
    rec_arr = []
    
    features_recommendations = recommender_by_features(song_name, data, number_recommended_by_features)
    for i in range (0, number_recommended_by_features):
        rec_arr.append(features_recommendations['track_name'].values[i] + '  -  ' + features_recommendations['artist_name'].values[i])
        song = features_recommendations['track_name'].values[i]
        data = data.drop(data[data['track_name'] == song].index)
      
    lyrics_recommendations = recomender_by_lyrics(song_name, data, number_recommended_by_lyrics)
    for i in range (0, number_recommended_by_lyrics):
        rec_arr.append(lyrics_recommendations['track_name'].values[i] + '  -  ' + lyrics_recommendations['artist_name'].values[i])

    return rec_arr

data = tracks_music

def get_all_songs_array(data=tracks):
    arr = []
    for idx, row in data.iterrows():
        song = row['track_name']
        artist = row['artist_name']
        song_artist = song + '  -  ' + artist
        arr.append((song_artist).upper())

    arr.sort()

    return arr

#user interface
import streamlit as st

st.title('Music recommendation system')
st.markdown('<p style="font-size:24px; font-style: italic;">Find new music to love</p>', unsafe_allow_html=True)

selection = st.selectbox('Choose a favourite song of yours:', get_all_songs_array(data))
song = selection.split('  -  ')[0].lower()


if st.button('Show me recommendations!'):
    recommendations = recommend_songs(song, data, 3, 2)
    st.write('Here are some songs you might enjoy:')
    st.markdown(f'### Recommended songs based on *{selection}*:')
    for i in range(0, 5):
        st.markdown(f'#### *{recommendations[i].upper()}*')

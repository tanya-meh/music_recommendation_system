# music_recommendation_system
Python content-based music recommendation system

Music Recommendation System

Discover your next favorite songs with a machine learning-based music recommendation system! This project utilizes data processing, vectorization, and similarity metrics to suggest tracks based on features or lyrics.


Features
- Recommend songs based on track features such as artist name, genre, and musical properties.
- Provide suggestions using lyrics similarity, powered by natural language processing (NLP).
- Two main recommendation approaches:
  - Feature-based recommendations
  - Lyrics-based recommendations
- User-friendly web interface built using Streamlit.


Dataset
The project uses the tcc_ceds_music.csv dataset. It includes information such as track names, artists, release dates, genres, and lyrics.


Code Highlights
- Data Preprocessing:
  - Dropping unnecessary columns and handling null or duplicate values.
  - Adding new features like release_decade and string_data.
- Feature-Based Recommendations:
  - Combines numeric and vectorized text features to calculate cosine similarity.
  - Recommends songs based on overall similarity.
- Lyrics-Based Recommendations:
  - Tokenizes and stems lyrics using NLTK.
  - Uses TF-IDF vectorization for text analysis.

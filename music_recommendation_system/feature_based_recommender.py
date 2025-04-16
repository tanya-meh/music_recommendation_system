from sklearn.metrics.pairwise import cosine_similarity
from decimal import Decimal
import numpy as np

def get_features_similarities(song_name, data, vectorizer):
    """
    Calculate similarity between the input song and all other songs in the dataset.
    """
    text_array1 = vectorizer.transform(data[data['track_name'] == song_name]['string_data']).toarray()
    num_array1 = data[data['track_name'] == song_name].select_dtypes(include=np.number).to_numpy()

    similarities = []
    for idx, row in data.iterrows():
        name = row['track_name']
        text_array2 = vectorizer.transform(data[data['track_name'] == name]['string_data']).toarray()
        num_array2 = data[data['track_name'] == name].select_dtypes(include=np.number).to_numpy()

        text_sim = Decimal(cosine_similarity(text_array1, text_array2)[0][0])
        num_sim = Decimal(cosine_similarity(num_array1, num_array2)[0][0])
        similarities.append(text_sim + num_sim)

    return similarities

def recommender_by_features(song_name, data, vectorizer, number_recommended=5):
    """
    Recommend songs based on feature similarity.
    """
    if number_recommended <= 0:
        return
    data['similarity_factor'] = get_features_similarities(song_name, data, vectorizer)
    data.sort_values(by=['similarity_factor'], ascending=False, inplace=True)
    return data[['track_name', 'artist_name']][1:number_recommended + 1]

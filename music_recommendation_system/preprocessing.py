import pandas as pd

def no_whitespace(text):
    """
    Remove the whitespace from a string.

    Args:
        text: Path to the CSV file.
    Returns:
        str: String with no whitespace.
    """
    res = str(text)
    return res.replace(' ', '')

def load_and_clean_data(file_path):
    """
    Load and preprocess the dataset.

    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    tracks = pd.read_csv(file_path)
    # Drop unneeded columns and rows with null values
    tracks = tracks.drop(['Unnamed: 0', 'len'], axis=1)
    tracks.dropna(inplace=True)
    tracks.drop_duplicates(subset=['track_name'], keep='first', inplace=True)

    # Convert all text columns to lowercase
    for column in ['artist_name', 'track_name', 'genre', 'lyrics', 'topic']:
        tracks[column].apply(lambda x: x.lower())

    # Add release_decade column
    tracks['release_decade'] = (tracks['release_date'] // 10) * 10
    tracks = tracks.drop(['release_date'], axis=1)

    # Add string_data column
    tracks['string_data'] = (tracks['artist_name'].apply(no_whitespace) + ' ' 
                         + tracks['genre'].apply(no_whitespace)  + ' ' 
                         + tracks['topic'].apply(no_whitespace))

    return tracks

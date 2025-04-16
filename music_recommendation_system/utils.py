def get_all_songs_array(data):
    """
    Get a sorted array of all songs and their respective artists.
    """
    arr = []
    for idx, row in data.iterrows():
        song = row['track_name']
        artist = row['artist_name']
        song_artist = song + '  -  ' + artist
        arr.append((song_artist).upper())

    arr.sort()

    return arr
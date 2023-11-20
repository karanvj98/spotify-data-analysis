import requests
import json

def curl_request(prefix, ID,access_token):
    
    #curl = pycurl.Curl()
    #command = "curl " + '"https://api.spotify.com/v1/playlists/' + playlist_id + '" -H "Authorization: Bearer ' + access_token + '"'
    #command = 'curl "https://api.spotify.com/v1/playlists/3cEYpjA9oz9GiPac4AsH4n" -H "Authorization: Bearer  BQDNq0JgmR88QTYgyeJQhX9Sr8J7KIQZmlJlASSjw39aQ1MszG6_jzoi4d_fMqze4dIqRL8GVb6zcalY8Xdr9myw8095RdATR2LWRDKovJg8CkVJb-U"'
    url = prefix + ID
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers)
    return response.json() 
    
    
def extract_track_features( ID ,is_playlist ):
    fh = open('access_token.txt','r')
    lines = fh.readlines()
    access_token = lines[0].split(':')[1].split('\n')[0]

    if is_playlist:
        prefix = 'https://api.spotify.com/v1/playlists/'
        response = curl_request(prefix, ID,access_token )
    else:
        prefix = 'https://api.spotify.com/v1/audio-features/'
        response = curl_request(prefix, ID,access_token )
   
    return response



if __name__ == '__main__':

    PLAYLIST_id = '3cEYpjA9oz9GiPac4AsH4n'
    response = extract_track_features(PLAYLIST_id, 1)
    PLAYLIST_name = response['name']

    items = response['tracks']['items'] 
    tracks = []
    playlist_dictionary = {}

    for i in range(0, len(items)): # each index contains a track
        track_dictionary = {}
        item = items[i]
        track_id = item['track']['id']
        SPOTIFY_album_id = item['track']['album']['id']
        ALBUM_name =  item['track']['album']['name']
        ALBUM_type =  item['track']['album']['album_type']
        RELEASE_date = item['track']['album']['release_date']

        artists = item['track']['artists']
        ARTIST_names = []
        #GENRES = []
        for j in range(0, len(artists)):
            artist =  artists[j]
            ARTIST_names.append(artist['name'])
            #print(artist['genres'] )
            #GENRES.append( artist['genres'] )
        
        DURATION_ms = item['track']['duration_ms']
        POPULARITY = item['track']['popularity']

        response = extract_track_features(track_id, 0)
        DANCEABILITY = response['danceability']
        ACOUSTICNESS = response['acousticness']

        ENERGY = response['energy']
        INSTRUMENTALNESS = response['instrumentalness']
        VALENCE = response['valence']
        TEMPO = response['tempo']
        LIVENESS = response['liveness']
        LOUDNESS = response['loudness']
        MODE = response['mode']
        KEY = response['key']
        SPEECHINESS = response['speechiness']

        track_dictionary = {
            'SPOTIFY_album_id': SPOTIFY_album_id,
            'ALBUM_name': ALBUM_name,
            'ALBUM_type':ALBUM_type,
            'ARTIST_names':ARTIST_names,
            #'GENRES':GENRES,
            'DURATION_ms':DURATION_ms,
            'RELEASE_date':RELEASE_date,
            'POPULARITY':POPULARITY,
            'DANCEABILITY':DANCEABILITY,
            'ACOUSTICNESS':ACOUSTICNESS,
            'ENERGY':ENERGY,
            'INSTRUMENTALNESS':INSTRUMENTALNESS,
            'VALENCE':VALENCE,
            'TEMPO':TEMPO,
            'LIVENESS':LIVENESS,
            'LOUDNESS':LOUDNESS,
            'MODE':MODE,
            'KEY':KEY,
            'SPEECHINESS':SPEECHINESS
        }
        tracks.append(track_dictionary)

    playlist_dictionary ={
        'PLAYLIST_id':PLAYLIST_id,
        'PLAYLIST_name':PLAYLIST_name,
        'TRACKS':tracks
    }




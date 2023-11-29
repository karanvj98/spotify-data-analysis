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
    
    
def extract_track_features( ID ,is_audio_feature ):
    fh = open('access_token.txt','r')
    lines = fh.readlines()
    access_token = lines[0].split(':')[1].split('\n')[0]

    if is_audio_feature:
        prefix = 'https://api.spotify.com/v1/audio-features/'
        response = curl_request(prefix, ID,access_token )
        print(response)

    else:
        prefix = 'https://api.spotify.com/v1/tracks/'
        response = curl_request(prefix, ID,access_token )
        print(response)
   
    return response



if __name__ == '__main__':

    with open('mpd.slice.0-999.json', 'r') as f:
        data = json.load(f)
        playlists = data['playlists']  
        for i in range(len(playlists)):
            print('playlist count:',i)
            playlist = playlists[i]
            PLAYLIST_id =  playlist['pid']
            PLAYLIST_name = playlist['name']
            tracks_list = playlist['tracks']
            tracks_features = []
            for j in range(len(tracks_list)):
                print('track count:',j)
                track_dictionary = {}
                track = tracks_list[j]
                track_id = track['track_uri'].split(':')[2].split('"')[0]
                track_features = extract_track_features(track_id,0)
                ALBUM_id = track['album_uri'].split(':')[2].split('"')[0]
                ALBUM_name = track['album_name']
                RELEASE_date = track_features['album']['release_date']
                ALBUM_type = track_features['album']['album_type']

                artists = track_features['artists']
                ARTIST_names = []
                for k in range(0, len(artists)):
                    artist =  artists[k]
                    ARTIST_names.append(artist['name'])
                
                DURATION_ms = track_features['duration_ms']
                POPULARITY = track_features['popularity']

                track_audio_features = extract_track_features(track_id,1)
                DANCEABILITY = track_audio_features['danceability']
                ACOUSTICNESS = track_audio_features['acousticness']

                ENERGY = track_audio_features['energy']
                INSTRUMENTALNESS = track_audio_features['instrumentalness']
                VALENCE = track_audio_features['valence']
                TEMPO = track_audio_features['tempo']
                LIVENESS = track_audio_features['liveness']
                LOUDNESS = track_audio_features['loudness']
                MODE = track_audio_features['mode']
                KEY = track_audio_features['key']
                SPEECHINESS = track_audio_features['speechiness']
                
                track_dictionary = {
                    'ALBUM_id': ALBUM_id,
                    'ALBUM_name': ALBUM_name,
                    'ALBUM_type':ALBUM_type,
                    'ARTIST_names':ARTIST_names,
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
                tracks_features.append(track_dictionary)



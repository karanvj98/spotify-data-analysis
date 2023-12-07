#!/usr/bin/env python
# coding: utf-8

# In[19]:


import requests
import json
import time
import pandas as pd

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
    batch_ID = ','.join(ID)
    if is_audio_feature:
        prefix = 'https://api.spotify.com/v1/audio-features?ids='
        response = curl_request(prefix, batch_ID,access_token )
        #print(response)

    else:
        prefix = 'https://api.spotify.com/v1/tracks?ids='
        response = curl_request(prefix, batch_ID,access_token )
        #print(response)
   
    return response



if __name__ == '__main__':
    with open('mpd.slice.0-999.json', 'r') as f:
        data = json.load(f)
        playlists = data['playlists']
        df_columns = ['PLAYLIST_id','PLAYLIST_name','TRACK_uri','TRACK_name','ALBUM_id', 'ALBUM_name', 'ALBUM_type', 'ARTIST_names', 'DURATION_ms', 'RELEASE_date', 'POPULARITY','DANCEABILITY', 'ACOUSTICNESS', 'ENERGY', 'INSTRUMENTALNESS', 'VALENCE', 'TEMPO', 'LIVENESS', 'LOUDNESS', 'MODE', 'KEY', 'SPEECHINESS']
        df = pd.DataFrame(columns=df_columns)
        for i in range(len(playlists)):
            print('playlist count:', i)
            playlist = playlists[i]
            PLAYLIST_id = playlist['pid']
            PLAYLIST_name = playlist['name']
            tracks_list = playlist['tracks']
            tracks_features = []
            if int(PLAYLIST_id) > 0:
                batch_size = 50
                for j in range(0, len(tracks_list), batch_size):
                    print("Batch",j)
                    batch_track_ids = [track['track_uri'].split(':')[2].split('"')[0] for track in tracks_list[j:j + batch_size]]
                    time.sleep(0.5)
                    print("Dataframe length", len(df))
                    album_batch_features = extract_track_features(batch_track_ids, 0)
                    audio_batch_features = extract_track_features(batch_track_ids, 1)
                    for k in range(len(tracks_list[j:j + batch_size])):
                        
                        artists = album_batch_features['tracks'][k]['album']['artists']
                        ARTIST_names = []
                        for a in range(0, len(artists)):
                            artist =  artists[a]
                            ARTIST_names.append(artist['name'])

                        track_dictionary = {
                            'ALBUM_id': album_batch_features['tracks'][k]['album']['uri'].split(':')[2].split('"')[0],
                            'ALBUM_name': album_batch_features['tracks'][k]['album']['name'],
                            'ALBUM_type': album_batch_features['tracks'][k]['album']['type'],
                            'ARTIST_names': ARTIST_names
,
                            'DURATION_ms': album_batch_features['tracks'][k]['duration_ms'],
                            'RELEASE_date': album_batch_features['tracks'][k]['album']['release_date'],
                            'POPULARITY': album_batch_features['tracks'][k]['popularity'],
                            'TRACK_uri': audio_batch_features['audio_features'][k]['uri'].split(':')[2].split('"')[0],
                            'TRACK_name': album_batch_features['tracks'][k]['name'],
                            'DANCEABILITY': audio_batch_features['audio_features'][k]['danceability'],
                            'ACOUSTICNESS': audio_batch_features['audio_features'][k]['acousticness'],
                            'ENERGY': audio_batch_features['audio_features'][k]['energy'],
                            'INSTRUMENTALNESS': audio_batch_features['audio_features'][k]['instrumentalness'],
                            'VALENCE': audio_batch_features['audio_features'][k]['valence'],
                            'TEMPO': audio_batch_features['audio_features'][k]['tempo'],
                            'LIVENESS': audio_batch_features['audio_features'][k]['liveness'],
                            'LOUDNESS': audio_batch_features['audio_features'][k]['loudness'],
                            'MODE': audio_batch_features['audio_features'][k]['mode'],
                            'KEY': audio_batch_features['audio_features'][k]['key'],
                            'SPEECHINESS': audio_batch_features['audio_features'][k]['speechiness'],
                            'PLAYLIST_id':  playlist['pid'],
                            'PLAYLIST_name': playlist['name']
                        }

                        tracks_features.append(track_dictionary)
                        df = pd.concat([df, pd.DataFrame(track_dictionary)], ignore_index=True)
                        
                
                
                


# In[ ]:


df.to_csv("intermediate_dataset", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





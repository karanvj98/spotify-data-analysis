# Karan

'''
Observations
   1) Note: ALBUM_type can be removed from dataset since all items are 'album  # len(pd.unique(df['ALBUM_type']))
   2) There is only one playlist with a single artist in our dataset creating issue for popularity_per_artist(). Does this needs to be done?
      Instead explored correlation between num_tracks_per_playlist v/s all features but it too has issue as mentioned in 3rd point
  3) Sum of Track count in each playlist not matching with the total no. of tracks
'''

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('merged_playlist_without_duplicates_0_999.csv')

# Karan
def best_features_for_popularity(data = df, k = 3):
    '''This method takes the 3 playlists with the highest median track popularity and finds which features are most correlated with popularity'''
    # Generate a heatmap with popularity vs. all features after finding the 3 playlists with the highest median popularity

    # Calculates a popularity value for each playlist as the median of the track popularities within each playlist
    popularity_per_playlist = df.groupby('PLAYLIST_id')['POPULARITY'].median()

    # Top k playlist popularity values
    top_k_playlist_popularities = popularity_per_playlist.nlargest(k)

    # Top k playlist ids as per playlist popularity
    top_k_popular_playlists = top_k_playlist_popularities.index.to_list()

    # Filtering features from dataframe corresponding to playlist ids of Top k popular playlists
    df_filtered = df[df['PLAYLIST_id'].isin(top_k_popular_playlists)]

    df_filtered = df_filtered[['PLAYLIST_name', 'TRACK_name', 'ALBUM_name', 'ARTIST_names', 'DURATION_ms', 'RELEASE_date', 'DANCEABILITY',
                      'ACOUSTICNESS', 'ENERGY', 'INSTRUMENTALNESS', 'VALENCE', 'TEMPO', 'LIVENESS', 'LOUDNESS', 'MODE', 'KEY', 'SPEECHINESS', 'POPULARITY']]

    # Calculates correlation of each track feature with popularity for the top k popular playlists
    features_correlation_with_popularity = df_filtered.corr()[['POPULARITY']].sort_values(by='POPULARITY', ascending=False)

    # Adjust the size of the heatmap
    plt.figure(figsize=(16, 6))

    # Plotting the correlation heatmap of each track feature v/s playlist popularity
    heatmap = sns.heatmap(features_correlation_with_popularity, vmin=-1, vmax=1, annot=True, center=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
    return heatmap

# Karan
def popularity_per_artist(data = df, k = 3):
    '''Find the playlists with only 1 artist and find the 3 with the most songs. Generate a heatmap comparing popularity vs. all features for these 3 playlists.'''
    tracks_per_playlist = df.groupby('PLAYLIST_id')['TRACK_name'].count()

    # Top k playlist with most songs
    top_k_playlist_with_most_songs = tracks_per_playlist.nlargest(3)

    # Top k playlist ids with most songs
    top_k_playlist_ids_with_most_songs = top_k_playlist_with_most_songs.index.to_list()

    # Filtering features from dataframe corresponding to playlist ids of Top k playlists with most songs
    df_filtered = df[df['PLAYLIST_id'].isin(top_k_playlist_ids_with_most_songs)]

    df_filtered = df_filtered[['PLAYLIST_name', 'TRACK_name', 'ALBUM_name', 'ARTIST_names', 'DURATION_ms', 'RELEASE_date', 'DANCEABILITY',
                      'ACOUSTICNESS', 'ENERGY', 'INSTRUMENTALNESS', 'VALENCE', 'TEMPO', 'LIVENESS', 'LOUDNESS', 'MODE', 'KEY', 'SPEECHINESS', 'POPULARITY']]

    # Calculates correlation of each track feature with count of songs for the top k playlists with most songs
    features_correlation_with_num_songs_per_playlist = df_filtered.corr()[['POPULARITY']].sort_values(by='POPULARITY', ascending=False)

    # Adjust the size of the heatmap
    plt.figure(figsize=(16, 6))

    # Plotting the correlation heatmap of each track feature v/s playlist popularity
    heatmap = sns.heatmap(features_correlation_with_num_songs_per_playlist, vmin=-1, vmax=1, annot=True, center=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
    return heatmap

"""#There is only one playlist with a single artist"""

# artists_per_playlist = df.groupby('PLAYLIST_id')['ARTIST_names'].nunique()
# artists_per_playlist.sort_values()

# playlists_with_single_artist = artists_per_playlist[artists_per_playlist == 1]
# playlists_with_single_artist

"""Sum of Track count in each playlist not matching with the total no. of tracks"""

tracks_per_playlist = pd.DataFrame(df.groupby('PLAYLIST_id')['TRACK_name'].count())
tracks_per_playlist.rename(columns = {'TRACK_name':'TRACK_count'}, inplace = True)
print(tracks_per_playlist)
# print(len(df))
# print(tracks_per_playlist['TRACK_count'].sum())
# print(df1.index[df1['TRACK_count'] == 51].tolist())

# Transform DataFrame tracks_per_playlist to replicate values based on their own value to get correlation 
tracks_per_playlist = tracks_per_playlist.loc[tracks_per_playlist.index.repeat(tracks_per_playlist['TRACK_count'])]
print(tracks_per_playlist)
# tracks_per_playlist['TRACK_count'].value_counts()[45]
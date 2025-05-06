# Analyzing Features that Shape Spotify Playlist Curation

## Project Overview:

This project investigates the crucial audio features that contribute to the cohesiveness of playlists. Data is obtained from the Spotify Million dataset using web-apis in batched manner and various audio features are analyzed. In analyzing the correlation, it has been observed that danceability, loudness, and acoustic-ness are highly interconnected. These elements significantly influence the overall feel and vibe of a playlist. Playlists designed to evoke happiness or excitement often feature songs with higher loudness and energy, reinforcing the idea that louder and more dynamic tracks contribute to a more upbeat and lively mood. Conversely, playlists curated to convey a sad or reflective mood are characterized by higher acousticness, indicating a preference for softer, more melodic, and less electronically influenced tracks in such playlists.

When considering the uniformity within playlists, danceability, tempo, and energy are the features that vary the least. This uniformity helps in maintaining a consistent mood or theme throughout the playlist. Interestingly, the popularity of a playlist does not necessarily depend on the audio features of the tracks it contains. It's more influenced by factors such as the frequency of plays and the recency of those plays, highlighting the impact of listener engagement and trends over the intrinsic qualities of the music. Additionally, the length of a playlist is determined by the type of songs it includes; longer playlists tend to have more acoustic songs and fewer party tracks. This suggests a preference for more relaxed and extended listening experiences in longer playlists, as opposed to the high-energy, shorter tracks typically found in party playlists. Leveraging the Spotify Million Playlist Dataset, we extract data through the Spotify API, aiming to identify significant features influencing playlist composition. The analysis is encapsulated in the code files ECE_143_final.ipynb and ECE_143_final.py, with results presented in Spotify_Data_Analysis_PPT_Group_12.pdf.


## Project Files:

- **mpd.slice.0-999.json:** Initial dataset containing playlist ID and track URI of 1000 playlists, downloaded from the Spotify Million Playlist Dataset.
  
- **access_token.txt:** Text file containing the access token required to call the Spotify API.
  
- **batch_data_extraction.ipynb:** Jupyter Notebook file for extracting data from the Spotify API.
  
- **batch_data_extraction.py:** Python script for extracting data from the Spotify API using track IDs mentioned in `mpd.slice.0-999.json`.
  
- **merged_playlist_without_duplicates_0_999.csv:** Dataset under investigation, resulting from the data extraction process.
  
- **ECE_143_final.ipynb:** Final project code in Jupyter Notebook format, including all analyses and conclusions.
  
- **ECE_143_final.py:** Final project code in Python script format, including all analyses and conclusions.
  
- **Spotify_Data_Analysis_PPT_Group_12.pdf:** Presentation slides summarizing the project.

## Third-Party Modules Used:

1. requests
2. json
3. time
4. pandas
5. numpy
6. os
7. seaborn
8. matplotlib
9. Spotify API

## Installing Third-Party Modules:

Use the following command to install the required third-party modules:

```bash
pip install requests json time pandas numpy os seaborn matplotlib
```

Use pip3 if python 3 is being used


## Steps to Generate Dataset "merged_playlist_without_duplicates_0_999.csv":

1. **Step 1:** Use the initial dataset, `mpd.slice.0-999.json`, containing information on 1000 playlists and associated data.

2. **Step 2:** Follow the document to generate an access token for the Spotify API: [Spotify API Access Token Guide](https://developer.spotify.com/documentation/web-api/concepts/access-token).

3. **Step 3:** Once the access token is generated, update the `access_token.txt` file with the obtained token.

4. **Step 4:** Run the `batch_data_extraction.ipynb` file to fetch the dataset using the Spotify API.

## Steps to Run the Script: `ECE_143_final.ipynb`

1. Use the dataset `merged_playlist_without_duplicates_0_999.csv` generated from the Spotify API.

2. Run the file.


## Analysis and Conclusions:

For an in-depth analysis and comprehensive conclusions drawn from the dataset, please refer to the Jupyter Notebook file [ECE_143_final_group12.ipynb](ECE_143_final_group12.ipynb). This notebook encompasses detailed exploratory data analysis, feature engineering, and insights gained from the Spotify Million Playlist Dataset. The findings shed light on the key audio features influencing playlist cohesiveness, providing valuable insights for music curation and automated playlist generation. Presentation slides are present in [Spotify_Data_Analysis_PPT_Group_12.pdf](Spotify_Data_Analysis_PPT_Group_12.pdf)

## References:

1. [Spotify Million Playlist Dataset Challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)

2. [Spotify Web API Documentation](https://developer.spotify.com/documentation/web-api)

## My Contribution

As part of this collaborative project, I was primarily responsible for:

- Analyzing audio features to explore what influences **playlist popularity**.
- Investigating how **playlist length** correlates with song types (e.g., acoustic vs. party tracks).
- Assisting with the **GitHub setup** and organizing the project structure.
- Supporting the creation of the **final presentation deck**.

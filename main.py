from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth
import spotipy 
import os
import pandas as pd
from tqdm import tqdm
import logging

# Load environment variables (.env)
load_dotenv('./.env')

# Authentication credentials
client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")
print(client_id, client_secret)
scope = "playlist-read-private"

def get_spotify_client():
    return spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                     client_secret=client_secret,
                                                     redirect_uri=redirect_uri,
                                                     scope=scope))

def get_user_playlists(sp):
    playlists = []
    results = sp.current_user_playlists()
    playlists.extend(results['items'])

    while results['next']:
        results = sp.next(results)
        playlists.extend(results['items'])

    return playlists

def get_playlist_tracks(sp, playlist_id):
    tracks = []
    results = sp.playlist_tracks(playlist_id)
    tracks.extend(results['items'])

    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])

    return tracks

def get_audio_features(sp, track_ids):
    audio_features = []
    for i in range(0, len(track_ids), 100):  # Process in chunks of 100
        chunk = track_ids[i:i+100]
        try:
            features = sp.audio_features(chunk)
            audio_features.extend(features)
        except spotipy.SpotifyException as e:
            logging.error(f"Failed to fetch audio features: {e}")
            break
    return audio_features

def extract_track_info_with_features(tracks, sp):
    track_list = []
    track_ids = [item['track']['id'] for item in tracks if item['track']]  # Get all track IDs

    audio_features = get_audio_features(sp, track_ids)

    for item, features in zip(tracks, audio_features):
        track = item['track']
        if track and features:  # Ensure both track and features are not None
            track_info = {
                'Track Name': track['name'],
                'Artist': track['artists'][0]['name'],
                'Album': track['album']['name'],
                'Release Date': track['album']['release_date'],
                'Track ID': track['id'],
                'URL': track['external_urls']['spotify'],
                'Danceability': features['danceability'],
                'Energy': features['energy'],
                'Key': features['key'],
                'Loudness': features['loudness'],
                'Mode': features['mode'],
                'Speechiness': features['speechiness'],
                'Acousticness': features['acousticness'],
                'Instrumentalness': features['instrumentalness'],
                'Liveness': features['liveness'],
                'Valence': features['valence'],
                'Tempo': features['tempo'],
                'Duration_ms': features['duration_ms'],
                'Time Signature': features['time_signature']
            }
            track_list.append(track_info)

    return track_list

def main():
    sp = get_spotify_client()

    playlists = get_user_playlists(sp)
    all_tracks = []

    for playlist in tqdm(playlists):
        print(f"Getting tracks from playlist: {playlist['name']}")
        tracks = get_playlist_tracks(sp, playlist['id'])
        all_tracks.extend(tracks)

    track_data_with_features = extract_track_info_with_features(all_tracks, sp)

    # Create a DataFrame
    df = pd.DataFrame(track_data_with_features)

    # Save DataFrame to CSV
    df.to_csv('playlist_tracks_with_features.csv', index=False)

    print("Tracks with audio features from all playlists have been saved to 'playlist_tracks_with_features.csv'.")

if __name__ == '__main__':
    main()

from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth
import spotipy
import os
import pandas as pd

# Load environment variables (.env)
load_dotenv()

# Authentication
client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

# Function to get spotipy.Spotify Object
def get_spotify_client(client_id, client_secret, redirect_uri, scope):
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                   client_secret=client_secret,
                                                   redirect_uri=redirect_uri,
                                                   scope=scope))
    return sp

# Function to get playlists
def get_playlists(sp):
    results = sp.current_user_playlists()
    for playlist in results['items']:
        print(f"Name: {playlist['name']}")
        print(f"Total tracks: {playlist['tracks']['total']}")
        print(f"Owner: {playlist['owner']['display_name']}")
        print(f"Playlist ID: {playlist['id']}")
        print('-' * 40)

# Function to get tracks from a specific playlist
def get_tracks_from_playlist(sp, playlist_id):
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
        
    track_ids = [item['track']['id'] for item in tracks]
    track_names = [item['track']['name'] for item in tracks]
    return track_ids, track_names

# Function to get followed artists
def get_followed_artists(sp, limit=20):
    results = sp.current_user_followed_artists(limit=limit)
    artists = results['artists']['items']
    
    while results['artists']['next']:
        results = sp.next(results['artists'])
        artists.extend(results['artists']['items'])
        
    for idx, artist in enumerate(artists):
        print(f"{idx + 1}. {artist['name']}")
        print(f"   Artist ID: {artist['id']}")
        print(f"   Genres: {', '.join(artist['genres'])}")
        print(f"   Followers: {artist['followers']['total']}")
        print(f"   Popularity: {artist['popularity']}")
        print('-' * 40)

def get_audio_analysis(sp, track_id):
    return sp.audio_analysis(track_id)

def get_audio_features(sp, track_ids):
    return sp.audio_features(track_ids)

def collect_audio_data(sp, playlist_id):
    track_ids, track_names = get_tracks_from_playlist(sp, playlist_id)
    
    analysis_data = []
    features_data = []
    
    for track_id in track_ids:
        analysis = get_audio_analysis(sp, track_id)
        features = sp.audio_features(track_id)[0]
        
        track_data = {
            'track_id': track_id,
            'track_name': track_names[track_ids.index(track_id)],
            'analysis': analysis,
            'features': features
        }
        
        analysis_data.append(analysis)
        features_data.append(features)
    
    return pd.DataFrame(features_data), pd.DataFrame(analysis_data)


# Authenticate and get Spotify client for playlists
# scope = 'playlist-read-private'
# sp = get_spotify_client(client_id, client_secret, redirect_uri, scope)
# print("User's Playlists:")
# get_playlists(sp)

# Authenticate and get Spotify client for followed artists
# scope = 'user-follow-read'
# sp = get_spotify_client(client_id, client_secret, redirect_uri, scope)
# print("User's Followed Artists:")
# get_followed_artists(sp)

# Authenticate and get Spotify client
scope = 'playlist-read-private user-read-recently-played user-read-playback-state'
sp = get_spotify_client(client_id, client_secret, redirect_uri, scope)

# Playlist ID
playlist_id = '37i9dQZF1DX9tPFwDMOaN1'

# Collect audio data
features_df, analysis_df = collect_audio_data(sp, playlist_id)

# Display the collected data
print("Audio Features DataFrame:")
print(features_df.head())

print("\nAudio Analysis DataFrame:")
print(analysis_df.head())
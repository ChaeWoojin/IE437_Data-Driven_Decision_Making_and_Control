from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth
import spotipy
import os
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class SpotifyDataCollector:
    def __init__(self, client_id, client_secret, redirect_uri):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.sp = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sp = None  # Explicitly releasing the Spotify client object

    def authenticate(self, scope):
        oauth = SpotifyOAuth(client_id=self.client_id,
                             client_secret=self.client_secret,
                             redirect_uri=self.redirect_uri,
                             scope=scope,
                             requests_timeout=30)  # Increased timeout to 30 seconds

        # Create a requests session
        session = requests.Session()
        retry = Retry(
            total=5,  # Total number of retries
            read=5,  # Number of read retries
            connect=5,  # Number of connection retries
            backoff_factor=0.3,  # A backoff factor to apply between attempts
            status_forcelist=(500, 502, 503, 504)  # A set of HTTP status codes to retry
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Pass the custom session to the Spotipy client
        self.sp = spotipy.Spotify(auth_manager=oauth, requests_session=session)

    def get_playlists(self):
        results = self.sp.current_user_playlists()
        for playlist in results['items']:
            print(f"Name: {playlist['name']}")
            print(f"Total tracks: {playlist['tracks']['total']}")
            print(f"Owner: {playlist['owner']['display_name']}")
            print(f"Playlist ID: {playlist['id']}")
            print('-' * 40)

    def get_tracks_from_playlist(self, playlist_id):
        results = self.sp.playlist_tracks(playlist_id)
        tracks = results['items']

        while results['next']:
            results = self.sp.next(results)
            tracks.extend(results['items'])
            
        track_ids = [item['track']['id'] for item in tracks]
        track_names = [item['track']['name'] for item in tracks]
        return track_ids, track_names

    def get_followed_artists(self, limit=20):
        results = self.sp.current_user_followed_artists(limit=limit)
        artists = results['artists']['items']
        
        while results['artists']['next']:
            results = self.sp.next(results['artists'])
            artists.extend(results['artists']['items'])
            
        for idx, artist in enumerate(artists):
            print(f"{idx + 1}. {artist['name']}")
            print(f"   Artist ID: {artist['id']}")
            print(f"   Genres: {', '.join(artist['genres'])}")
            print(f"   Followers: {artist['followers']['total']}")
            print(f"   Popularity: {artist['popularity']}")
            print('-' * 40)

    def get_liked_tracks(self):
        results = self.sp.current_user_saved_tracks()
        tracks = results['items']
        
        while results['next']:
            results = self.sp.next(results)
            tracks.extend(results['items'])
            
        track_ids = [item['track']['id'] for item in tracks]
        track_names = [item['track']['name'] for item in tracks]
        return track_ids, track_names

    def get_audio_analysis(self, track_id):
        return self.sp.audio_analysis(track_id)

    def get_audio_features(self, track_ids):
        return self.sp.audio_features(track_ids)

    def collect_audio_data(self):
        track_ids, track_names = self.get_liked_tracks()
        
        features_data = self.get_audio_features(track_ids)
        analysis_data = [self.get_audio_analysis(track_id) for track_id in track_ids]
        
        # Add track names to the features data
        for i in range(len(features_data)):
            features_data[i]['track_name'] = track_names[i]
        
        features_df = pd.DataFrame(features_data)
        
        # Flatten analysis data and add track IDs and names
        analysis_flattened = []
        for i, analysis in enumerate(analysis_data):
            flattened = {
                'track_id': track_ids[i],
                'track_name': track_names[i],
                'duration': analysis['track']['duration'],
                'tempo': analysis['track']['tempo'],
                'time_signature': analysis['track']['time_signature']
            }
            
            if analysis['bars']:
                flattened['bars_start'] = analysis['bars'][0]['start']
                flattened['bars_duration'] = analysis['bars'][0]['duration']
                flattened['bars_confidence'] = analysis['bars'][0]['confidence']
            
            if analysis['beats']:
                flattened['beats_start'] = analysis['beats'][0]['start']
                flattened['beats_duration'] = analysis['beats'][0]['duration']
                flattened['beats_confidence'] = analysis['beats'][0]['confidence']
            
            if analysis['sections']:
                flattened['sections_start'] = analysis['sections'][0]['start']
                flattened['sections_duration'] = analysis['sections'][0]['duration']
                flattened['sections_confidence'] = analysis['sections'][0]['confidence']
                flattened['sections_loudness'] = analysis['sections'][0]['loudness']
                flattened['sections_tempo'] = analysis['sections'][0]['tempo']
                flattened['sections_tempo_confidence'] = analysis['sections'][0]['tempo_confidence']
                flattened['sections_key'] = analysis['sections'][0]['key']
                flattened['sections_key_confidence'] = analysis['sections'][0]['key_confidence']
                flattened['sections_mode'] = analysis['sections'][0]['mode']
                flattened['sections_mode_confidence'] = analysis['sections'][0]['mode_confidence']
                flattened['sections_time_signature'] = analysis['sections'][0]['time_signature']
                flattened['sections_time_signature_confidence'] = analysis['sections'][0]['time_signature_confidence']
            
            if analysis['segments']:
                flattened['segments_start'] = analysis['segments'][0]['start']
                flattened['segments_duration'] = analysis['segments'][0]['duration']
                flattened['segments_confidence'] = analysis['segments'][0]['confidence']
                flattened['segments_loudness_start'] = analysis['segments'][0]['loudness_start']
                flattened['segments_loudness_max'] = analysis['segments'][0]['loudness_max']
                flattened['segments_loudness_max_time'] = analysis['segments'][0]['loudness_max_time']
                flattened['segments_loudness_end'] = analysis['segments'][0]['loudness_end']
                flattened['segments_pitches'] = analysis['segments'][0]['pitches']
                flattened['segments_timbre'] = analysis['segments'][0]['timbre']
            
            if analysis['tatums']:
                flattened['tatums_start'] = analysis['tatums'][0]['start']
                flattened['tatums_duration'] = analysis['tatums'][0]['duration']
                flattened['tatums_confidence'] = analysis['tatums'][0]['confidence']
            
            analysis_flattened.append(flattened)
        
        analysis_df = pd.DataFrame(analysis_flattened)
        
        return features_df, analysis_df

    def collect_audio_data_from_playlist_id(self, playlist_id):
        track_ids, track_names = self.get_tracks_from_playlist(playlist_id)
        
        analysis_data = []
        features_data = []
        
        for track_id in track_ids:
            analysis = self.get_audio_analysis(track_id)
            features = self.sp.audio_features(track_id)[0]
            features['track_name'] = track_names[track_ids.index(track_id)]
            analysis_data.append(analysis)
            features_data.append(features)
        
        return pd.DataFrame(features_data), pd.DataFrame(analysis_data)

if __name__ == "__main__":
    # Load environment variables (.env)
    load_dotenv()

    # Authentication
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

    # Use context manager to ensure cleanup
    with SpotifyDataCollector(client_id, client_secret, redirect_uri) as spotify_collector:
        # Authenticate and get User's Saved Albums
        scope = 'user-library-read'
        spotify_collector.authenticate(scope)

        # Collect audio data from liked tracks
        features_df, analysis_df = spotify_collector.collect_audio_data()
        features_df.to_csv('./features_df.csv', index=False, encoding='utf-8')
        analysis_df.to_csv('./analysis_df.csv', index=False, encoding='utf-8')

        # Uncomment the following to collect audio data from a playlist by ID
        # scope = 'playlist-read-private user-read-recently-played user-read-playback-state'
        # spotify_collector.authenticate(scope)
        # playlist_id = '37i9dQZF1

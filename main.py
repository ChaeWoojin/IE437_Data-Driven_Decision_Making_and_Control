from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from requests import post, get
import spotipy.util as util
import spotipy
import os
import json
import base64

# Load environment variables (.env)
load_dotenv()

# Authentication
client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")
scope = 'playlist-read-private'

# Access Token
def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
    
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials", "scope": scope, "redirect_uri": redirect_uri}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token

# create header whenever request needed
def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

# Input: artist_name, Output: artist_information
def search_for_artist(token, artist_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    query = f"?q={artist_name}&type=artist&limit=1"

    query_url = url + query
    result = get(query_url, headers=headers)
    json_result = json.loads(result.content)["artists"]["items"]
    
    if len(json_result) == 0:
        print("No artist with this name exists")
        return None

    return json_result[0]

# Input: artist_id, Output: corresponding artist's tracks
def get_songs_by_artist(token, artist_id):
    url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks?country=US"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)["tracks"]
    return json_result

    
# Get current user's playlists
def get_playlists():
    playlists = sp.current_user_playlists()
    for playlist in playlists['items']:
        print(f"Name: {playlist['name']}")
        print(f"Total tracks: {playlist['tracks']['total']}")
        print(f"Owner: {playlist['owner']['display_name']}")
        print(f"Playlist ID: {playlist['id']}")
        print('-' * 40)



if __name__ == '__main__':
    # Non
    # token = get_token()

    # Initialize Spotipy with user authorization
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                   client_secret=client_secret,
                                                   redirect_uri=redirect_uri,
                                                   scope=scope))

    # Fetch and display playlist information
    get_playlists()


    # Get Title of Songs
    # result = search_for_artist(token, "ACDC")
    # artist_id = result["id"]
    # songs = get_songs_by_artist(token, artist_id)
    
    # for idx, song in enumerate(songs):
    #     print(f"{idx + 1}. {song['name']}")

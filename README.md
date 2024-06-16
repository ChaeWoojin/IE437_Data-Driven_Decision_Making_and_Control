
# Optimizing Contextual Bandit Algorithms for Personalized Music Recommendation

## Project Overview

This project aims to compare and optimize various contextual bandit algorithms for personalized music recommendation. By leveraging different bandit strategies, we can enhance the recommendation system's ability to balance exploration and exploitation, ultimately improving user satisfaction with music recommendations. The project focuses on algorithms such as LinUCB, Thompson Sampling, Epsilon Greedy, UCB1, Softmax, and BayesUCB.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Algorithms](#algorithms)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used in this project is a Spotify track dataset containing various features that describe the characteristics of each track. The detailed description of the features is as follows:

- **Like**: float - 1 denotes the track that the user likes, and 0 otherwise.
- **Danceability**: float - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
- **Energy**: float - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
- **Key**: int - The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
- **Loudness**: float - The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 dB.
- **Mode**: int - Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
- **Speechiness**: float - Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audiobook, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-spee...
- **Acousticness**: float - A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
- **Instrumentalness**: float - Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
- **Liveness**: float - Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
- **Valence**: float - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
- **Tempo**: float - The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
- **Duration_ms**: int - The duration of the track in milliseconds.
- **Time_signature**: int - An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).
- **Year**: int - Year describes when the track was released.
- **Popularity**: int - The popularity of the artist. The value will be between 0 and 100, with 100 being the most popular. The artist's popularity is calculated from the popularity of all the artist's tracks.
- **Explicit**: int - Whether or not the track has explicit content (1 = yes, 0 = no or unknown).

## Algorithms

The project implements and compares the following contextual bandit algorithms:

- **LinUCB**: Uses linear models to incorporate contextual information.
- **Thompson Sampling**: Uses Bayesian methods to balance exploration and exploitation.
- **Epsilon Greedy**: Randomly explores with probability `epsilon` and exploits the best-known action otherwise.
- **UCB1**: Uses upper confidence bounds to select actions.
- **Softmax**: Uses a softmax function to assign probabilities to actions based on their estimated rewards.
- **BayesUCB**: Combines Bayesian methods with UCB for action selection.

## Hyperparameter Tuning

Hyperparameter tuning is performed for each algorithm to find the optimal parameters that minimize cumulative regret and maximize the winning rate. The key hyperparameters tuned include:

- **LinUCB**: `alpha` (exploration parameter)
- **Thompson Sampling**: (no tunable parameters in this basic implementation)
- **Epsilon Greedy**: `epsilon` (exploration probability)
- **UCB1**: (no tunable parameters in this basic implementation)
- **Softmax**: `tau` (temperature parameter)
- **BayesUCB**: `alpha` (parameter controlling the confidence bound)

## Results

The results are compared based on cumulative regret and winning rate. The algorithms are evaluated on their ability to recommend tracks that align with the user's preferences, balancing exploration of new tracks and exploitation of known liked tracks.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ChaeWoojin/IE437_Data-Driven_Decision_Making_and_Control.git
   cd IE437_Data-Driven_Decision_Making_and_Control
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the comparison script:**
   ```bash
   # cd bandit_models 
   python comparison.py

   # cd candidates 
   python comparison.py
   ```

2. **Visualize the results:**
   - The script will display plots comparing the cumulative regret for different algorithms with their best-tuned hyperparameters.
   - Winning rates for each algorithm will be printed in the console.

## Contributing

We welcome contributions to enhance this project. Please fork the repository, create a new branch, and submit a pull request with your changes. Ensure your code follows the existing style and includes tests where applicable.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

# IE437 Data-Driven Decision Making and Control

**Topic: Recommend Music For Woojin!**

Given several quantitative features of music from Spotify (https://developer.spotify.com/documentation/web-api), the goal of this project is to learn my preferences in an statistical manner so that learn about myself.

Data: tracks' feature labeled with my preferences (1 for the likes, 0 for the remainders)


Online Learning Procedure (Version 1):

1. Data are provided in an unit of batches, which contains 10 data of tracks.

2. Decision maker decides each track to recommend or not according to the classifier algorithm; Loss are evaluated, and history are updated [feature, loss, ...]

3. After each episode, model are recalibrated according to the history.

4. Calculate the test data regret

Online Learning Procedure (Version 2):

1. Random policy suggests any 10 music(slates) at first.

2. According to the score(cumulative preference value), model learns my personal statistical distribution over features according to the loss

3. Model keep acquiring my data, recalibrating itselves.

4. At final, recommend tracks from undiscovered playists.


Although it is online algorithm, before start, I labeled every data.


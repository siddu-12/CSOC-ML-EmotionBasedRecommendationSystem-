# CSOC-ML-EmotionBasedRecommendationSystem 
This project basically involves detecting emotions from the video of a user using OpenCV and predicting songs from Spotify corresponding to the users mood from the emotions detected.

To predict songs based on user moods I decided to clusters my liked songs into 6 playlists corresponding to 6 emotions- "Fear, Happy, Angry, Surprised, Sad and Neutral". To do so I had to accomplish the following tasks:

->**Acquire Data**- I used data from the Spotify Dataset 1921–2020 found on Kaggle. Spotify Dataset 1921–2020 contains more than 160,000 songs collected from Spotify Web API, and also you can find data grouped by artist, year, or genre in the data section.(I have uploaded the dataset here as well for reference.) It provides a large variety of features; however, I used 8 features for describing a song i,e acousticness, danceability, liveness, energy, instrumentalness, loudness, speechiness and valence as I found them to be of more use while detecting the mood of a song.

->**Build a clustering model**- I ended up using the K-Means Clustering Algorithm, which is used to determine distributions in data. It is an unsupervised learning algorithm that groups similar data points into k groups by calculating distances to centroids. To attain that goal it looks for a predefined number (k) of clusters. Since I wanted to cluster my songs into 6 playlists corresponding to 6 emotions I chose to set the number k to 6 here although the elbow plot favoured setting it to 4 or 5 to be the optimal choice.

->**Find out an appropriate classifier and train on the data acquired**- Considering that I had now obtained labels as our clusters, I could now easily implement a classification algorithm that would help classify my saved songs on Spotify. Furthermore, it would allow us to classify recommended songs and separate them into different playlists to serve the purpose of our problem. In my notebook we can find four models compared in terms of accuracy score, which are K-Neighbors Classifier, Random Forest Classifier, Support Vector Classifier and Naive Bayes. Support Vector Classifier(with kernel set to linear) turned out to be the best model in terms of accuracy score, which made up roughly 0.988, hence I went ahead with using it for future classification.

->**Classify my songs and separate them into playlists**- Spotify’s API provides a set of useful functions for this purpose.I was able to obtain a dataset of all the songs I've liked. I then went ahead to classify my liked songs using the classifier mentioned above. Finally, I sorted my songs into 6 different playlists representing these categories.

[Cluster 0](https://open.spotify.com/playlist/6yvDfcDh1my9pIXWGYLq2k?si=be35a1842c3b40e0) has high values of danceability and low values of speechiness and valence and could correspond to songs one might listen to when feeling fear and wants to divert their mind with high tempo song.

[Cluster 1](https://open.spotify.com/playlist/2WXT06lbhbWWvaJJ7WGAlK?si=aa483296d94b469c) has a low value of valence which correponds to negative emotions in the song and low values of liveliness and danceability as well. Thus it represents songs one might listen to when in a sad mood.

[Cluster 2](https://open.spotify.com/playlist/0lrgUJi282zwjJkN5KGFVq?si=49454821c8f84944) has high values of energy and liveness but low value of valence thus it can correspond to songs one might listen to when in an angry mood.

[Cluster 3](https://open.spotify.com/playlist/5am2EI25FgGFZ514PEQf6b?si=35e500a4d1464adf) has high danceability, liveliness and energy values. It also has a high valence value which depicts a positive uplifiting song. This cluster corresponds to songs one can listen to when happy.

[Cluster 4](https://open.spotify.com/playlist/6CCV3TdQxbxriDQ4qr8Fcd?si=833c4f84b72b42ac) has unusually high loudness and median values for other features. It can include songs one might want to listen to when they wish to be surprised or dont know what to listen to.

[Cluster 5](https://open.spotify.com/playlist/1XGeqvSdPF8MZmj6VyZby8?si=9d1f996edd874208) has high values of instrumentalness and low values of speechiness and danceability, this can correpond to neutral emotions.

The next half of the project involved developing a detector to determine the mood of the user from a video of their face. Still an amateur at Computer Vision, I went ahead with using The Face Emotion Recognizer (generally knowns as the FER) which is an open-source Python library built and maintained by Justin Shenk for this purpose as it is widely used for sentiment analysis of images and videos. The project is built on a version that uses a convolution neural network with weights mentioned in the HDF5 data file present in the [source code](https://github.com/justinshenk/fer/tree/master/src/fer/data) of this system’s creation model. This can be overridden by using the FER constructor when the model is called and initiated.

1. **MTCNN (multi cascade convolutional network)** is a parameter of the constructor. It is a technique to detect faces. When it is set to ‘True’ the [MTCNN model](https://towardsdatascience.com/robust-face-detection-with-mtcnn-400fa81adc2e) is used to detect faces, and when it is set to ‘False’ the function uses the default [OpenCV Haarcascade classifier](https://towardsdatascience.com/face-detection-with-haar-cascade-727f68dafd08).

2. **detect_emotions()**: This function is used to classify the detection of emotion and it registers the output into six categories, namely, ‘fear’, ‘neutral’, ‘happy’, ’sad’, ‘anger’, and ‘disgust’. Every emotion is calculated, and the output is put on a scale of 0 to 1.

The program starts by taking into input the image or video that needs analysis. The FER() constructor is initialized by giving it a face detection classifier (either Open CV Haarcascade or MTCNN). We then call this constructor’s detect emotions function by passing the input object (image or video) to it. The result achieved is an array of emotions with a value mentioned against each. Finally, the ‘top_emotion’ function can seclude the highest valued emotion of the object and return it.

Check out the model's output for a classic Micheal Scott gif xD.

![happy-emotional](https://user-images.githubusercontent.com/96650742/186596589-b178e2c8-dfe7-4b11-a363-3648be831352.gif) ![Graph](https://user-images.githubusercontent.com/96650742/186598719-3614eb21-cf2a-4c9c-8de0-7d1d5fd5d6c2.png)

Once I had obtained the emotion of the user and detected their mood I now simply had to use the Spotify API to suggest songs from the playlist I had already created corresponding to the respective mood. Here's what the model came up with for the above gif:
![Recommendations](https://user-images.githubusercontent.com/96650742/186599065-ed47a82a-2ce0-417c-819b-bac67df794d9.png)

You can find the results for a couple more samples in the results folder. I had a great experience making this project wherein I got to learn about the real time applications of CV and the working of a recommendation system in detail and would love to build on similar such ideas in the future.

Possible improvements possible in this project could be deploying this model as a web-app that can take real time video data as input and allow the user to sign in to their spotify account and classify and recommend music from their liked songs based on the mood detected.


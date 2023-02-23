## Introduction  
  From Spotify Wrapped to Discover Weekly, Spotify is renowned for its recommendation algorithm, with more than 81% of Spotify listeners listing personalization or discoverability as their favorite feature [[1]](#references). Genre classification is crucial in this recommendation process since genres can represent shared properties like ambience, lyrics, composition, etc. that comprise a user’s musical preferences. As such, Spotify uses machine learning methods such as convolutional neural networks (CNN) to analyze both the raw audio files and metadata such as “loudness” or “danceability” to determine a song’s genre [2]. Others have also conducted feature extraction and deep learning on lyrics of songs in order to classify genres with some success [3].

## Problem Definition  
  Despite a large body of research, music genre classification remains a challenging task. One of the fundamental difficulties is the subjective nature of music genres, which varies across cultures and time periods. Additionally, the wide variety of sub-genres and hybrid genres makes it difficult to create a thorough and reliable classification system. 
  The incentive of the project is to provide such a classification system for music industry professionals to target and promote their music to specific audiences, and to better identify stylistic trends and influences.  
  The dataset we use, collected by the original author through Spotify API, contains audio features of the top 2000 tracks of all time on Spotify [4]. These features are: genre, year, bpm, energy, danceability, loudness (dB), valence, length, acousticity, speechiness, and popularity.

## Method  

**K-nearest neighbors**  
  Using KNN requires a dataset of music track with their respected genres. Using the Spotify API, we have 11 different features not including genres. In order to train the model, we need to find a value for $k$, and a distance metric. The value of $k$  needs to be at least the number of genres, and the distance metric to be Euclidean distance. To evaluate the accuracy, we plan to compare the predicted and true genres.  
  To test if a value of $k$ higher than the number of genres is better, we are going to conduct cross-validation. We choose a range of values of $k$. For each value in that range, train a different KNN model using the same training set. Evaluate each model’s performance on the test set and record the evaluation metrics. The better value of $k$ would be one that produced the optimal performance on the validation set.   

## Project Contributors

| Member | Contributions |
| --- | ----------- |
| Ruwei Ma | Problem Definition, Methods, Github Pages |
| Annette Gisella | Introduction, Github Pages, References, Problem Definition |
| Richard Hoepfinger | Presentation, Problem Definition, Methods, Team Communication |
| Tuan Ha | Potential Results/Discussion, Methods |
| Arthur Odom | Gantt Chart, Data Collection |

## Gantt Chart

## References
1.  https://newsroom.spotify.com/2022-06-08/spotify-shares-our-vision-to-become-the-worlds-creator-platform/  
2.  https://d3.harvard.edu/platform-digit/submission/spotify-how-data-is-used-to-enhance-your-listening-experience/#:~:text=Using%20CNN%2C%20Spotify%20analyzes%20raw,further%20optimize%20its%20recommendation%20engine.  
3.  A. Kumar, A. Rajpal and D. Rathore, "Genre Classification using Feature Extraction and Deep Learning Techniques," 2018 10th International Conference on Knowledge and Systems Engineering (KSE), Ho Chi Minh City, Vietnam, 2018, pp. 175-180, doi: 10.1109/KSE.2018.8573325. https://ieeexplore.ieee.org/document/8573325  
4. (dataset) https://www.kaggle.com/datasets/iamsumat/spotify-top-2000s-mega-dataset/discussion/246253?resource=download

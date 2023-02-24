# Music Genre Classification
## Introduction  
  From Spotify Wrapped to Discover Weekly, Spotify is renowned for its recommendation algorithm, with more than 81% of Spotify listeners listing personalization or discoverability as their favorite feature [[1]](#references). Genre classification is crucial in this recommendation process since genres can represent shared properties like ambience, lyrics, composition, etc. that comprise a user’s musical preferences. As such, Spotify uses machine learning methods such as convolutional neural networks (CNN) to analyze both the raw audio files and metadata such as “loudness” or “danceability” to determine a song’s genre [[2]](#references). Others have also conducted feature extraction and deep learning on lyrics of songs in order to classify genres with some success [[3]](#references).

## Problem Definition  
Despite this, music genre classification remains a challenging task because of the subjective nature of music genres, which varies across cultures and time periods. Additionally, the wide variety of sub-genres and hybrid genres makes it difficult to create a thorough and reliable classification system. 

The goal of this project is to explore the use of deep learning in genre classification using Spotify metadata. Unlike classical machine learning methods, deep learning is better suited to capture complexities in large amounts of data, allowing for more comprehensive and extensive classification of musical genres [5]. 

We will create our own dataset by making requests to the Spotify API [6]. Features we will gather include: ID, title, artist, genre, year, bpm, energy, danceability, loudness (dB), valence, length, acousticity, speechiness, and popularity. We aim to obtain a balanced representation of around 10 genres, each having at least 500 data points.


## Method  

**Convolutional Neural Networks**  

We will majorly use convolutional neural networks (CNN) for our project. We will first preprocess our data by normalizing the numeric features to a universal scale. Then, we split the data into training, validation, and testing sets. Outliers will also be identified in preprocessing using DB-SCAN.

We will subsequently build the CNN model. This process involves selecting an appropriate architecture, and specifying the number and size of the layers. We might use techniques such as dropout or batch normalization to improve performance and prevent overfitting.

The next step is to train the model using the training set. In this stage, the model will adjust its weights to minimize the categorical cross-entropy loss over the training set, penalizing the model for assigning low probability to the true label.

After training the model, we will evaluate its performance on the validation and testing sets. If unsatisfactory, we will tune its hyperparameters, and retrain the model until the desired performance is achieved. 

## Potential results and Discussion

A general baseline can be created for each of the general genres of music (pop, rock, hip-hop, etc). The baseline contains a standard value for each of the features, and a song can be tested to see how likely it matches with that genre. Songs that match to a genre can be used for the audience of that genre. 

In addition to matching songs to the genre baseline, trends can also be discovered. The year and popularity of each song is recorded and those metrics can reveal trends in the music industry.


Metrics from sickit learn : 
- Accuracy_score : this can be used to verify how accurate the classification model is in correctly identifying the music genres across the whole sample
  - Return : number of correct positives / whole sample
- Recall_score : this can be used to show how many genre classifications were predicted correctly out of all correct samples
  - Return : true positives / (true positives + false negatives)
- Precision_score : this can be used to reveal how many genre matches were actually of the correct genre, and was not a false positive
  - Return : true positives / (true positives + false positives)


## Project Contributors

| Member | Contributions |
| --- | ----------- |
| Ruwei Ma | Problem Definition, Methods, Github Pages |
| Annette Gisella | Introduction, Github Pages, References, Problem Definition |
| Richard Hoepfinger | Presentation, Problem Definition, Methods, Team Communication |
| Tuan Ha | Potential Results/Discussion, Methods |
| Arthur Odom | Gantt Chart, Data Collection |

## Gantt Chart

[Hyperlink to Google Drive](https://docs.google.com/spreadsheets/d/1l8K8Aj34vmP7cY6OAWpWX1UIrCkKELlk/edit?usp=sharing&ouid=110632432805448997773&rtpof=true&sd=true)

![Screenshot of Gaant Chartchart for those who don't like links](/gaantChartScreenshot.png "A screenshot for your convenience")

All M2 tasks have 6 days extra, all M3 tasks have 14 days. This is so the schedule doesn't explode the minutes we miss a deadline.

## References
1.  “Spotify Shares Our Vision to Become the World’s Creator Platform.” Spotify, 8 June 2022, [newsroom.spotify.com/2022-06-08/spotify-shares-our-vision-to-become-the-worlds-creator-platform/.](https://newsroom.spotify.com/2022-06-08/spotify-shares-our-vision-to-become-the-worlds-creator-platform/)
2.  Tebuev, Alan. “Spotify - How Data Is Used to Enhance Your Listening Experience.” Digital Innovation and Transformation, Harvard Business School, 27 Mar. 2022, [d3.harvard.edu/platform-digit/submission/spotify-how-data-is-used-to-enhance-your-listening-experience/#:~:text=Using%20CNN%2C%20Spotify%20analyzes%20raw.](https://d3.harvard.edu/platform-digit/submission/spotify-how-data-is-used-to-enhance-your-listening-experience/#:~:text=Using%20CNN%2C%20Spotify%20analyzes%20raw,further%20optimize%20its%20recommendation%20engine)  
3.  A. Kumar, A. Rajpal and D. Rathore, "Genre Classification using Feature Extraction and Deep Learning Techniques," 2018 10th International Conference on Knowledge and Systems Engineering (KSE), Ho Chi Minh City, Vietnam, 2018, pp. 175-180, doi: [10.1109/KSE.2018.8573325.](https://ieeexplore.ieee.org/document/8573325)
4. (dataset) https://www.kaggle.com/datasets/iamsumat/spotify-top-2000s-mega-dataset/discussion/246253?resource=download

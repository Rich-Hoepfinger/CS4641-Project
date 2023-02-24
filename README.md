# Music Genre Classification
## Introduction  
More than 81% of Spotify listeners list personalization or discoverability as their favorite feature [[1]](#references). Genre classification is crucial in this recommendation process since genres can represent shared features that comprise a user’s musical preferences. As such, Spotify analyzes both the raw audio files and metadata such as “loudness” or “danceability” to determine a song’s genre [[2]](#references). Classifiers such as K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Random Forests have commonly been used to conduct genre classification using the metadata [[3]](#references). Others have also attempted converting raw audio files into spectrogram images to be fed into a convolutional neural network (CNN) [[5]](#references).

## Problem Definition  
Despite this, music genre classification remains a challenging task because of the subjective nature of music genres, varying across cultures and time periods, and the wide variety of sub and hybrid genres.

The goal of this project is to employ deep learning in genre classification using Spotify metadata. Unlike classical machine learning methods, deep learning is better suited to capture complexities in large amounts of data, allowing for more thorough and reliable classification of musical genres [[5]](#references). 

We will create our own dataset by making requests to the Spotify API [[6]](#references). Features we will gather include: ID, title, artist, genre, year, bpm, energy, danceability, loudness (dB), valence, length, acousticity, speechiness, and popularity. We aim to obtain a balanced representation of around 10 genres, each having at least 500 data points.

## Method  

**Convolutional Neural Networks**  

We will mainly use convolutional neural networks (CNN) for our project: 
1. Preprocess the data by normalizing the numeric features to a universal scale. Identify outliers using DB-SCAN. Then, the data is split into training, validation, and testing sets.
2. Build the CNN model. Select an appropriate architecture, and specify the number and size of the layers. We might use techniques such as dropout to prevent overfitting.
3. Train the model. Here, the model will adjust its weights to minimize the categorical cross-entropy loss over the training set, penalizing low probability assignments to the true labels.
4. Evaluate the model’s performance on the validation and testing sets. If unsatisfactory, we will tune its hyperparameters and retrain the model until the desired performance is achieved. 


## Potential results and Discussion

A general baseline can be created for each genre, containing a standard value for each of the features. A song can be tested against these baselines to see how likely it matches with that genre. In addition to this, trends can also be discovered. The year and popularity of each song is recorded and those metrics can reveal trends in the music industry.

Metrics from scikit learn : 
* Accuracy_score : number of correct positives / whole sample  
verifies how accurate the classification model is in correctly identifying the music genres across the whole sample
* Recall_score :  true positives / (true positives + false negatives)  
shows how many genre classifications were predicted correctly out of all correct samples
* Precision_score : true positives / (true positives + false positives)  
reveals how many genre matches were actually of the correct genre, and was not a false positive

## Project Contributors

| Member | Contributions |
| --- | ----------- |
| Ruwei Ma | Problem Definition, Methods, Github Pages |
| Annette Gisella | Introduction, Github Pages, References, Problem Definition |
| Richard Hoepfinger | Presentation, Methods, Team Communication |
| Tuan Ha | Potential Results/Discussion, Methods |
| Arthur Odom | Gantt Chart, Data Collection |

## Gantt Chart

<iframe width="700" height="500" frameborder="0" src="https://docs.google.com/spreadsheets/d/1l8K8Aj34vmP7cY6OAWpWX1UIrCkKELlk/edit?usp=sharing&ouid=110979405002483791203&rtpof=true&sd=true" title="description"></iframe>

[Hyperlink to Google Drive](https://docs.google.com/spreadsheets/d/1l8K8Aj34vmP7cY6OAWpWX1UIrCkKELlk/edit?usp=sharing&ouid=110632432805448997773&rtpof=true&sd=true)

![Screenshot of Gaant Chartchart for those who don't like links](./gaantChartScreenshot.png "A screenshot for your convenience")

All M2 tasks have 6 days extra, all M3 tasks have 14 days. This is so the schedule doesn't explode the minutes we miss a deadline.

## References

1. [Spotify. (2022, June 8). Spotify Shares Our Vision To Become the World’s Creator Platform. Spotify Newsroom.](https://newsroom.spotify.com/2022-06-08/spotify-shares-our-vision-to-become-the-worlds-creator-platform/)
2. [Tebuev, A. (2022, March 27). Spotify - How data is used to enhance your listening experience. Digital Innovation and Transformation; Harvard Business School.](https://d3.harvard.edu/platform-digit/submission/spotify-how-data-is-used-to-enhance-your-listening-experience/)
3. [Luo, K. (2018). Machine Learning Approach for Genre Prediction on Spotify Top Ranking Songs. https://doi.org/10.17615/j9m1-tz22/](https://doi.org/10.17615/j9m1-tz22/)
4. [Poonia, Sahil & Verma, Chetan & Malik, Nikita. (2022). Music Genre Classification using Machine Learning: A Comparative Study. 13. 15-21.](https://www.researchgate.net/publication/362619781_Music_Genre_Classification_using_Machine_Learning_A_Comparative_Study/)
5. [Wolfewicz, A. (2022, April 21). Deep learning vs. machine learning – What’s the difference? Levity.](https://levity.ai/blog/difference-machine-learning-deep-learning/)
6. [Spotify. (2019). Web API. Spotify for Developers.](https://developer.spotify.com/documentation/web-api/)

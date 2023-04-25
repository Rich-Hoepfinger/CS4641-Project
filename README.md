# Music Genre Classification
## Introduction  
More than 81% of Spotify listeners list personalization or discoverability as their favorite feature [[1]](#references). Genre classification is crucial in this recommendation process since genres can represent shared features that comprise a user’s musical preferences. As such, Spotify analyzes both the raw audio files and metadata such as “loudness” or “danceability” to determine a song’s genre [[2]](#references). Classifiers such as K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Random Forests have commonly been used to conduct genre classification using the metadata [[3]](#references). Others have also attempted converting raw audio files into spectrogram images to be fed into a convolutional neural network (CNN) [[5]](#references).

## Problem Definition  
Despite this, music genre classification remains a challenging task because of the subjective nature of music genres, varying across cultures and time periods, and the wide variety of sub and hybrid genres. The goal of this project is to employ deep learning and K-nearest neighborss in genre classification using Spotify metadata.

## Method  
We will use **neural networks**, **K-nearest neighbors (KNN)**, **decision trees**, and **support vector machines (SVM)** for our project: 
1. Preprocess the data by normalizing the numeric features to a universal scale. Identify outliers using DB-SCAN. Then, the data is split into training, validation, and testing sets.
2. Build the model(s). Select an appropriate architecture, and specify the number and size of the layers. We might use techniques such as dropout to prevent overfitting.
3. Train the model(s). Here, the model will adjust its weights to minimize the categorical cross-entropy loss over the training set, penalizing low probability assignments to the true labels.
4. Evaluate and compare the model(s)’ performance on the testing set. If unsatisfactory, we will tune its hyperparameters and retrain the model until the desired performance is achieved. 

### Data Collection and Duplicate Removal
First, the team created a comprehensive list of genres to be used as classification labels. The generated list consists of 12 genres: Pop, Hip Hop, Rock, Country, Dance/Electronic, R&B, Metal, Jazz, Reggae, Disco, Folk, Orchestral.

The next step is to generate Spotify playlists of around 500 tracks for each of the chosen genres. Each genre was assigned to a team member who would be responsible for compiling tracks that belong to that genre. The created playlists can be found here: [Pop](https://open.spotify.com/playlist/13Cm1hem9RE4v2ZOMJv34T?si=C9qRRpOQT4-tIqZlzERd6w&app_destination=copy-link), [Hip Hop](https://open.spotify.com/playlist/5K9FlaF7V8Ib4X09rl23w6?si=XnauqM5BTiiqvp88B1bDwA&app_destination=copy-link), [Rock](https://open.spotify.com/playlist/5Wk9TcVaNE5yCyli90HmaR?si=7e1c58897b224157), [Country](https://open.spotify.com/playlist/6hjSKEoPqPLPTt3u0e9bLQ?si=0c844e194b8e467f), [Electronic](https://open.spotify.com/playlist/4ZMSlQbw13hExG4ztynNNV?si=1176e787add3496d), [R&B](https://open.spotify.com/playlist/5WJXKWvPeWA9ubmvVsMTBh?si=YQ0Hnol7Q6S7sivtxHAssA&app_destination=copy-link), [Metal](https://open.spotify.com/playlist/2mDXGVXMG4ZMQaR26iZVC7?si=2beae20751ab4364), [Jazz](https://open.spotify.com/playlist/6GrLcuf2cf8g6h8lkZ0h7H?si=3b460ce380ff4ee6), [Reggae](https://open.spotify.com/playlist/6E5Fr9QU6yAety6Y6pw11u?si=28e0d5583c814bb0), [Disco](https://open.spotify.com/playlist/15X96mpCbP1ZiX8WIBqOhO?si=ba46abbe1146480f), [Folk](https://open.spotify.com/playlist/5fEekkUMaM2Le4FL38UuKx?si=b569e2ad8a264b6b), [Orchestral](https://open.spotify.com/playlist/6HGXewRDu4sH2qy8NYnNba?si=e41039b4ee764d3d).

Next, using the Spotify API and the spotipy library in Python, metadata describing each track’s popularity and audio features was extracted from the Spotify API to be used as features in the classification model. A description of each of these features obtained from the Spotify API [[6]](#references) is portrayed below:

| Feature | Variable Type | Description |
| --- | ----------- | ---------------|
| Popularity | numeric int | The popularity of the album. The value will be between 0 and 100, with 100 being the most popular. |
| Danceability | numeric float | A measure from 0.0 to 1.0. How suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. |
| Energy | numeric float | A measure from 0.0 to 1.0. Represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. |
| Key | categoric int | The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1. |
| Loudness | numeric float | The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db. |
| Mode | categoric int | Indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0. |
| Speechiness | numeric float | A measure from 0.0 to 1.0. Detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.|
| Acousticness | numeric float | A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. |
| Instrumentalness | numeric float | A measure from 0.0 to 1.0. Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. |
| Liveliness | numeric float | A measure from 0.0 to 1.0. Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. |
| Valence | numeric float | A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). |
| Tempo | numeric float | The overall estimated tempo of a track in beats per minute (BPM). |
| Duration | numeric int | The duration of the track in milliseconds. |
| Time Signature | categoric int | An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of "3/4", to "7/4". |

This information was then compiled into a dataframe, with additional information such as the track name and the main artist. A label was also assigned to each track based on which genre playlist it belongs to. Afterwards, the data was further cleaned by checking for duplicates in the data. There were some cases where duplicates of songs occurred because Spotify may have the same track in different albums. These duplicates were removed from the dataset. There were also cases where the same track or song appeared in more than one genre playlist. This was to be expected since many songs do not fit perfectly into a single genre. To resolve this, a definitive genre was decided for each conflict after listening to the track. As a result of these methods, 134 data points were removed, resulting in a final dataframe of 6100 data points. Each data point was also checked for missing values, and none were found. 

### Removing Outliers with DBSCAN
Outliers decrease the statistical power of tests we conduct during feature reduction and in training the model. Therefore, DBScan, a clustering algorithm, was used to identify outliers, which were then removed prior to data normalization. 

For outlier detection, it’s important to make sure that the outliers identified are truly anomalous and not sheerly extreme values that are important to genre detection. While DBScan worked on the dataset relatively easily, the difficulty lay in finding a good Epsilon and number of minimum points. These 2 parameters will be kept as hyperparameters for all models and be tuned together with the rest. 

DB-SCAN was tested using several values for epsilon and min points to generate the following graph:
 
![Bubble graph of removed songs per Epsilon & Min_Points](Graphs/euclidean.png)
 
In the graph above, the size of the circles represent the number of outliers identified to be removed. From this, a reasonable value to use as a starting point for epsilon and minimum points was determined to be 5 and 4, respectively.

These values resulted in 1 cluster and removed 37 songs when applied to the dataset.

### Exploratory Data Visualization
Below are some visualizations of the data with outliers removed:
<iframe src="graphdisplay.html" width="832" height="519"></iframe>
<iframe src="Graphs/piechart.html" width="832" height="519"></iframe>

Based on the above pie chart, the removal of 37 songs did not significantly affect the equal representation of genres.

### Normalization
Our dataset consists of numeric and categorical variables. Categorical variables, like key, time signature should be kept as is. Numeric variables are mostly already min-max normalized by Spotify. The only ones left unnormalized are popularity, tempo, and duration(ms). 

3 ways to normalize the data were proposed.
* Min-max: (X- Xmin)/(X_max - X_min). This ensures 0-1 scale. 
* Z-score: (X - mean)/std. By one sigma from mean, ~68% of data are between 0-1.
* Softmax: softmax(x_i) = e^(x_i) / sum_j(e^(x_j)). A probability distribution for each class.

For each method, we normalize all numeric variables. It turned out that Z-score was the only normalization method commendable due to higher performance on top of the DB-scanned dataset. Hence, Z-score was chosen as the normalization method for all the generated models.

### Feature Reduction

In order to reduce the time and space complexity of the model(s), feature reduction was conducted. The first step was to check for multicollinearity, and thus the correlation matrix between variables was examined.

<iframe src="Graphs/CorrelationMatrix_zscore.html" width="832" height="519"></iframe>

Using this correlation matrix, features were dropped until all correlation coefficients fell below a specified threshold. It was important to determine the right correlation threshold, as removing a variable could also cause the model performance to worsen. As a starting point, a threshold of 0.6 was selected. Under this threshold of 0.6, there were 3 pairs of variables to fix: Loudness and energy, having correlation of 0.81; energy and acousticness, having correlation of -0.78;  loudness and energy, correlation of -0.69. A decision was made to remove loudness and energy to resolve the 3 pairs. As a result, all collinearity decreased to below 0.6.

<iframe src="Graphs/CorrelationMatrix_removed.html" width="832" height="519"></iframe>

### Model Exploration

Metrics from scikit learn : 
* Accuracy_score : number of correct positives / whole sample  
verifies how accurate the classification model is in correctly identifying the music genres across the whole sample
* Recall_score :  true positives / (true positives + false negatives)  
shows how many genre classifications were predicted correctly out of all correct samples
* Precision_score : true positives / (true positives + false positives)  
reveals how many genre matches were actually of the correct genre, and was not a false positive

As outlined in the methods, the supervised learning models we chose to use were Neural Networks, K-Nearest Neighbors, Decision Trees, and Support Vector Machines. All models were trained on ~4500 songs, and it was tested against ~1500 songs. 

The genres were converted to labels on an interval of 0 to 11. These are the label representations : 

| Label | Genre |
| ---- | ------ |
| 0 | R&B |
| 1 | Country | 
| 2 | Disco |
| 3 | Electronic |
| 4 | Folk |
| 5 | HipHop |
| 6 | Jazz |
| 7 | Metal | 
| 8 | Orchestral |
| 9 | Pop |
| 10 | Reggae | 
| 11 | Rock | 

**Neural Network (Multi-layer Perceptron)**  : 

After removing outliers from the original data set, the following is the model’s prediction alongside its metrics.

<iframe src="model/images/dbscan/dbNN.png" width="832" height="519"></iframe>

Accuracy : 52.6%
Precision : 52.3%
Recall : 52.1%

According to the metrics, about 52.6% of the songs were predicted correctly out of ~1500 songs. There were a large number of songs incorrectly predicted with no clear pattern of how similar genres were to each other.  Believing that the data set could be made cleaner, the collinearity between features were found as an attempt to remove conflicting features. After reducing the number of features, the following is model prediction and metrics. 

<iframe src="model/images/collinearity/colNN.png" width="832" height="519"></iframe>

Accuracy: 52.2 %
Precision: 51.7%
Recall: 51.8%

Using the reduced data set resulted in similar results. After poor classification from the Neural Net, there was a pivot to using KNN. The reasoning was that the Neural Net may not have had enough data to accurately predict genres.

**K-Nearest Neighbors** : 
Using the data set without outliers from performing DBscan, the following is the model prediction and metrics. 

<iframe src="model/images/dbscan/dbKNN.png" width="832" height="519"></iframe>

Accuracy : 55.6%
Precision : 57.6% 
Recall : 54.9%

There was roughly a 3% increase in accuracy as compared to the Neural Network on the same data set. However, there is a distinct pattern. Many of the mislabeled predictions allude to the genre of country music. Besides country music, the mislabeled predictions are much more consistent. 

Using the data set with reduced features, the following is the model prediction and metrics. 

<iframe src="model/images/collinearity/colKNN.png" width="832" height="519"></iframe>


Accuracy : 54% 
Precision : 54.7% 
Recall : 53.5%

Although the accuracy was lower, the incorrect predictions align much stronger with country music. After K-Nearest Neighbors, a decision tree classifier was attempted to gain an attempt at improved accuracy.

**Decision Tree** :
Using the data set with reduced features, the following is the model prediction amd metrics.

<iframe src="model/images/collinearity/dt_3.png" width="832" height="519"></iframe>

Accuracy : 45% 
Precision : 44.9% 
Recall : 45.2%

After attempting to use Decision Trees as an alternative model, the accuracy was noticeable lower than using a neural network whether that be Multi-Layer Perceptron or K-Nearest Neighors. The accuracy is lower, which suggests that the model is still overfitting some data. Support Vector Machines can help mediate overfitting through the regularization parameter, C. Hence, we attempted SVM on the dataset.

**Support Vector Machine**:
Using the data set with reduced features, the following is the model prediction and metrics. 

<iframe src="model/images/collinearity/SVM_2.png" width="832" height="519"></iframe>

Accuracy : 59.9% 
Precision : 58.9% 
Recall : 59%

This model performs considerably better than the other explored models. Hence, we decided to further pursue this model. To improve the model's performance, hyperparameter tuning was conducted.

### Hyperparameter Tuning

For each of the following parameters, different values (indicated in brackets) were iterated through to obtain the highest testing accuracy.

**DB-SCAN**
* minPts [range from 1-10]- minimum number of points in neighborhood in DB-SCAN
* epsilon [range from 1-100] - radius of hypersphere in DB-SCAN
* distance ["euclidean", "cosine", "cityblock", "l1", "l2", "hamming"] - the distance function used in DB-SCAN

**Feature Reduction**
* th [0.6, 0.8] - correlation threshold for feature reduction by mutlicollinearity analysis

**Support Vector Machine**
* kernel [‘poly’, ‘rbf’, ‘sigmoid’] - the kernel type used to transform the data
* degree [range from 1-10] - if kernel = 'poly', specifies the degree of the polynomial, else changes nothing
* C [range from 0.1-2.5] - indicates how much incorrect classifications should be penalized
* break_ties [true, false] - if true, break ties according to confidence values

## Results and Discussion

After tuning, here are the optimal performance metrics and hyperparameters for each model.

**Decision Tree**
* Removed Songs: 7
* Epsilon: 19
* Min_Points: 2
* Metric: l1
* Colinearity Threashold: 0.8
* Dropped Columns: ['loudness']
* Model: Decision Tree
* Accuracy: 0.503
* Precision: 0.499
* Recall: 0.497

**Neural Networks**
* Removed Songs: 9
* Epsilon: 13
* Min_Points: 9
* Metric: euclidean
* Colinearity Threashold: 0.8
* Dropped Columns: ['loudness']
* Model: Neural Net
* Accuracy: 0.56
* Precision: 0.554
* Recall: 0.552

**SVM**
* Removed Songs: 42
* Epsilon: 15
* Min Points: 3
* Metric: cityblock
* Colinearity Threshold: 0.8
* Dropped Columns: ['loudness"]
* Model: SVM
* Accuracy: 0.6256
* Precision: 0.6342
* Recall: 0.6343
* C: 2.3
* Kernel: rbf
* Degree: 3
* Break Ties: False

The optimal model in all 3 metrics is SVM. To reduce the randomness of DBScan results, we ran 10 replications of the DBScan on SVM to tune epsilon and mininum points, with constant C, kernal, degree and break ties. In the following image we plotted the model accuracy against different values of $\epsilon$ at the optimal minimum points. 

![eps_10avg](https://user-images.githubusercontent.com/106047524/234424208-6de56890-7859-4fb4-9699-873de80b0791.png)

There is a clear maximum at $\epsilon = 9$. In an attempt to verify the stability of our result, 10 replications were ran again. And we achieved maximum at a different value of $\epsilon$.

![eps_10avg_2](https://user-images.githubusercontent.com/106047524/234424235-a98585f2-dd53-4866-887c-0bc565153086.png)

The diagram look much different in the 2nd experiment and produces a different optimal $\epsilon$. Since DBScan relies heavily on the initial points, to reduce the randomness incurred by the initialisation, it was decided to increase the replication number to 50. All points will be selected as starting points eventually as replication number goes to infinity. That way we reduce the effect of initialization on a certain point. 

![eps_avg_50](https://user-images.githubusercontent.com/106047524/234424259-4040c436-82e2-4495-a151-0a0d304650b9.png)

50 replications gave maximum accuracy at $\epsilon = 13$. The crests and troughs are present still. This is a caveat that the dataset may not be 

![minpts_avg_50](https://user-images.githubusercontent.com/106047524/234424278-91dfc38c-cc97-4a84-911c-46fb4fdb514c.png)


Confusion Matrix

![confusionMatrix](https://user-images.githubusercontent.com/106047524/234429329-cfcfa193-e7ec-46f4-8cb1-c1561dd2174b.png)


## Project Contributors

| Member | Contributions |
| --- | ----------- |
| Ruwei Ma | Exploratory Hyperparameter Tuning, Normalisation, Feature Reduction, Problem Definition, Discussion |
| Annette Gisella | Report Write-Up, Data Collection, Duplicate Removal, Feature Reduction, Introduction, Discussion |
| Richard Hoepfinger | Hyperparameter Tuning, Exploratory Data Visualization, Team Communication, Presentation, Discussion |
| Tuan Ha | Hyperparameter Tuning, Neural Network, KNN, Results, Discussion |
| Arthur Odom | Gantt Chart, DBSCAN, Discussion |

## Gantt Chart

<iframe width="1080" height="920" frameborder="0" allowfullscreen="true" src="https://docs.google.com/spreadsheets/d/1l8K8Aj34vmP7cY6OAWpWX1UIrCkKELlk/edit?usp=sharing&ouid=110979405002483791203&rtpof=true&sd=true" title="description"></iframe>

[**Full Gantt Chart**](https://docs.google.com/spreadsheets/d/1l8K8Aj34vmP7cY6OAWpWX1UIrCkKELlk/edit?usp=sharing&ouid=110632432805448997773&rtpof=true&sd=true)
![Gantt Chart Overview](./gaantChartScreenshot.png "Gantt Chart Overview")

All M2 tasks have 6 days extra, all M3 tasks have 14 days. This is so the schedule doesn't explode the minutes we miss a deadline.

## References

1. [Spotify. (2022, June 8). Spotify Shares Our Vision To Become the World’s Creator Platform. Spotify Newsroom.](https://newsroom.spotify.com/2022-06-08/spotify-shares-our-vision-to-become-the-worlds-creator-platform/)
2. [Tebuev, A. (2022, March 27). Spotify - How data is used to enhance your listening experience. Digital Innovation and Transformation; Harvard Business School.](https://d3.harvard.edu/platform-digit/submission/spotify-how-data-is-used-to-enhance-your-listening-experience/)
3. [Luo, K. (2018). Machine Learning Approach for Genre Prediction on Spotify Top Ranking Songs. https://doi.org/10.17615/j9m1-tz22/](https://doi.org/10.17615/j9m1-tz22/)
4. [Poonia, Sahil & Verma, Chetan & Malik, Nikita. (2022). Music Genre Classification using Machine Learning: A Comparative Study. 13. 15-21.](https://www.researchgate.net/publication/362619781_Music_Genre_Classification_using_Machine_Learning_A_Comparative_Study/)
5. [Wolfewicz, A. (2022, April 21). Deep learning vs. machine learning – What’s the difference? Levity.](https://levity.ai/blog/difference-machine-learning-deep-learning/)
6. [Spotify. (2019). Web API. Spotify for Developers.](https://developer.spotify.com/documentation/web-api/)

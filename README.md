# Subreddit Classification Using Web APIs and NLP

## Problem Statement
Using NLP and classification techniques, can a model be created that outperforms our baseline model when predicting which of two subreddits a post came from? More specifically, can this be done from the titles of two subreddits that focus on videos, gifs, and images rather than text?

## Executive Summary

The goal of this project was to use [Pushshift's Reddit API](https://github.com/pushshift/api) to extract and analyze the titles of posts from two subreddits and, using a variety of NLP and classification techniques, determine if a model can accurately predict which subreddit a post is from. 

Our chosen subreddits to compare are:

- [r/AnimalsBeingBros](https://www.reddit.com/r/AnimalsBeingBros/)
- [r/AnimalsBeingJerks](https://www.reddit.com/r/AnimalsBeingJerks/)

Both of these subreddits define themselves as, "a place for sharing videos, gifs, and images," which is perfect for assessing if a model can be created that can outperform our baseline model when predicting which of two non text-heavy subreddits a post came from.

The following sections are supported by the respective numbered Jupyter Notebooks:

[01: Data Collection & Cleaning](https://github.com/ksylvia16/Subreddit-Classification/blob/b0f6cf7ffb5ef25ab921d9bb4ac126fb3127637d/code/01_Data_Collection_%26_Cleaning.ipynb)

The data used to train and test our model sets comes from Reddit's API, which returns .json dictionaries for data requests. We iterated through to collect as many posts as permitted, and obtained from 6500 posts each from r/AnimalsBeingBros and r/AnimalsBeingJerks from the beginning of each month starting in January 2016 and ending in June 2021. These were parsed into subreddit and title content and saved to csv using a Pandas DataFrame. Data Cleaning included concatenating the dataframe, removing null values, and dropping duplicate titles. 

[02: EDA & Pre-Processing](https://github.com/ksylvia16/Subreddit-Classification/blob/6a53fbc8137e25cf36102460211f89122cde5301/code/02_EDA_%26_Pre-Processing.ipynb)

Initial EDA included analyzing title length distribution by word and character count in each subreddit, analyzing the top 20 words used in each subreddit, and running a Sentiment Intensity Analysis to determine the average positive, negative, and neutral polarity scores in each subreddit. Pre-processing includes tokenizing using Regex to remove punctuation and contractions, as well as stemming, lemmatizing, and vectorizing with Count Vectorizer and TF-IDF (Term Frequency-Inverse Document Frequency) to enhance modeling response. 

[03: Model Selection and Evaluation](https://github.com/ksylvia16/Subreddit-Classification/blob/6a53fbc8137e25cf36102460211f89122cde5301/code/03_Model%20Selection_%26_Evaluation.ipynb)

First, we divide our data into a Train and Test vector to begin modeling. Using Pipeline and GridSearchCV hyperparameter optimization, we selected 8 models to build and examine:
- Logistic Regression
- Bernouli Naive Bayes
- Multinomial Naive Bayes
- Random Forest
- Gradient Boost
- AdaBoost
- Support Vector Machine
- K-Nearest Neighbors

Once the best model was determined, hyperparameter tuning continued to optimize our final model. The Multinomial Naive Bayes model performed best with a 73.3% Testing Score.

# Data Dictionary

| Feature              | Type     | Description                                                          |
|----------------------|----------|----------------------------------------------------------------------|
| stemmed              | *object* | String of full stemmed title, tokens separated by spaces             |
| is_AnimalsBeingJerks | *int*    | Binary Variable: 0 for r/AnimalsBeingBros, 1 for r/AnimalsBeingJerks |


# Recommendations & Conclusions

**Recommendations to Improve Our Model**

The first recommendation to improve our model would be to collect more data. Since both of these subreddits are not text heavy, more posts should be collected. Second, another recommendation would be to include comments as a feature. At just an average of 6 words in each title, adding comments to our data would likely improve our model. My last recommendation would be to investigate more parameters.
As wonderful as Pipelines and GridSearchCV are, using them in practice can be incredibly time consuming. More extensive searches with different parameters can be conducted to further optimize our model.

**Conclusions**

With an average of just 6 words in each title to make predictions on, the model performed well. With many one worded, emoji, and slang titles, classifying these titles into their correct subreddits would be a task even humans would find difficult to do. The question remains answered; with a 45.7% improvement from our baseline model, yes, a model can outperform our baseline model when predicting which of two non text-heavy subreddits a post came from.

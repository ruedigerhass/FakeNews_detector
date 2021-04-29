# Fake News Detector

## Detect Fake News in BuzzFeed articles with machine learning algorithms

### Data Sience Capstone Project with the [Neue Fische](https://www.neuefische.de/) school and pool for digital talent.

This project is still in progress.

![](https://cdn.aerzteblatt.de/bilder/2020/05/img243412729.jpg)

## Never trust the internet!

## Objective

Build a machine learning model that classifies news articles into Fake News and not Fake News content (reliable news) by Natural Language Processing.

## Methods

I decided to go with pretrained representations of Bag-of-Words, Word2Vec and [DistilBERT](https://huggingface.co/distilbert-base-multilingual-cased). As a baseline model I chose logistic regression as a binary classifier. Furthermore, I deploeyed an recurrent neural network, particularly LSTM model, which is considered best suited for NLP tasks.

## Data

The data comes from the facebook page of digital media BuzzFeed and provides 1469 political articles of 9 publishers in a week close to the US elections 2016. They are fact-checked by professional journalists at BuzzFeed and can be downloaded [here](https://github.com/BuzzFeedNews/2016-10-facebook-fact-check/tree/master/data)

## Data Science Lifecycle

The work process follows the data science lifecycle with adjustments:

  - Business Understanding
  - Data Minig
  - Data Cleaning
  - Data Exploration
  - Feature Engineering
  - Predictive Modeling
  - Data Visualization

## Python Modules

Pandas / NumPy / Matplotlib / Seaborn / NLTK / Sklearn / Tensorflow / Keras / Bag-of Words / Word2Vec / DistilBERT

## Future Work

Implementing an more efficent RNN and collect more and better data.

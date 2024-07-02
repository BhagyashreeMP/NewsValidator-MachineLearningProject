# NewsValidator-MachineLearningProject
NewsValidator-MachineLearningProject is a machine learning tool for detecting fake news. It uses advanced algorithms and NLP to analyze and verify news articles and social media content, aiming to improve media literacy and ensure reliable information.

Train.csv: https://drive.google.com/file/d/1XA-QiUoEZ4-sR18sWQuM7OOyZGvzuouU/view?usp=sharing

This repository contains a machine learning project aimed at detecting fake news using a RandomForestClassifier. The project leverages text data from news articles to build a binary classification model capable of distinguishing between real and fake news.

Table of Contents
Project Overview
Features
Installation
Usage
Data Preprocessing
Model Training and Evaluation
Results
Contributing
License
Project Overview
Fake news has become a significant concern, contributing to misinformation and public distrust. This project aims to create a machine learning model to automatically classify news articles as fake or real. By preprocessing the text data and training a RandomForestClassifier, we achieve effective results in detecting fake news.

Features
Data Preprocessing: Combines author and title into a single text feature, performs text cleaning, and applies stemming.
Feature Extraction: Uses TF-IDF vectorization to convert text data into numerical vectors suitable for machine learning models.
Model Training: Implements a RandomForestClassifier to classify news articles.
Model Evaluation: Evaluates model performance using accuracy scores and classification reports.
Installation
To run this project, follow these steps:

Clone the Repository:

git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
Install Dependencies: Make sure you have Python 3.6+ and install the required packages.

pip install -r requirements.txt
Download NLTK Stopwords:

import nltk
nltk.download('stopwords')
Usage
Prepare the Dataset: Ensure your dataset is in CSV format with columns for author, title, and label. Place it in the /content/ directory as train.csv.

Run the Script: Execute the script to preprocess data, train the model, and evaluate it.

python news_validator.py
Evaluate Results: Check the terminal or output files for accuracy scores and classification reports.

Data Preprocessing
The data preprocessing steps involve:

Loading Data: Reading the dataset into a pandas DataFrame.
Handling Missing Values: Filling missing values with empty strings.
Combining Text Features: Merging author and title into a new content column.
Text Cleaning and Stemming: Removing non-alphabet characters, converting text to lowercase, removing stop words, and applying stemming.
Model Training and Evaluation
Feature Extraction:

Using TF-IDF vectorization to transform text data into numerical features.
Model Training:

Training a RandomForestClassifier on the preprocessed and vectorized text data.
Evaluation:

Evaluating model performance with accuracy scores and classification reports on both training and testing datasets.
Results
The project demonstrates effective performance in detecting fake news, with high accuracy and detailed classification metrics. The RandomForestClassifier is robust, providing reliable predictions on unseen data.

Feel free to reach out for any questions or suggestions regarding this project. We hope this repository provides a helpful starting point for developing robust fake news detection models.

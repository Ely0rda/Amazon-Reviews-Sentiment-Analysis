# Amazon Reviews Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-green.svg)](https://www.nltk.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0.3-yellow.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

> NLP-based sentiment analysis of Amazon fine food reviews to classify customer opinions and extract insights.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Techniques](#analysis-techniques)
- [Results](#results)
- [Skills Demonstrated](#skills-demonstrated)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

## Overview

This project applies Natural Language Processing techniques to analyze Amazon Fine Food Reviews and extract sentiment information. By examining customer reviews, the system determines whether customers express positive, negative, or neutral opinions about products. This analysis provides valuable insights for businesses to understand customer satisfaction and product perception.

The implementation uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer, a lexicon and rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media and customer reviews.

## Dataset

The project uses the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset, which contains:

- 568,454 food reviews from Amazon
- Review text and metadata (ratings, helpfulness votes)
- Data spans approximately 10 years
- Review scores from 1-5 stars

## Tech Stack

- **Python**: Primary programming language
- **NLTK**: Natural Language Toolkit for NLP operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## Key Features

1. **Text Preprocessing**: Tokenization and preprocessing of review text data
2. **Sentiment Analysis**: VADER-based sentiment scoring for each review
3. **Distribution Analysis**: Analysis of sentiment distribution across review scores
4. **Entity Recognition**: Named Entity Recognition to identify product names and organizations
5. **Visualization**: Graphical representation of sentiment patterns and trends

## Screenshots

![Sentiment Distribution](https://example.com/sentiment_distribution.png)
*Example sentiment distribution visualization*

## Architecture

The project follows a data pipeline architecture:

1. **Data Loading**: Import and initial preprocessing of the Amazon reviews dataset
2. **Text Processing**: NLP preprocessing including tokenization and part-of-speech tagging
3. **Sentiment Analysis**: Application of VADER sentiment analyzer
4. **Result Aggregation**: Calculation of sentiment metrics and statistics
5. **Visualization**: Generation of insightful visualizations based on analysis results

## Installation

1. Clone the repository:

```bash
git clone https://github.com/username/Amazon-Reviews-Sentiment-Analysis.git
cd Amazon-Reviews-Sentiment-Analysis
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Download NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
```

## Usage

1. Launch Jupyter Notebook:

```bash
jupyter notebook
```

2. Open `notebook.ipynb` to explore the analysis.

3. Example code for sentiment analysis:

```python
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Analyze a sample text
sample = "This product is excellent and I highly recommend it!"
sentiment_scores = sia.polarity_scores(sample)
print(sentiment_scores)
```

## Analysis Techniques

The project employs several NLP techniques:

- **Tokenization**: Breaking down reviews into individual tokens (words, punctuation)
- **Part-of-Speech Tagging**: Identifying grammatical elements (nouns, verbs, adjectives)
- **Named Entity Recognition**: Extracting organizations, brands, and product names
- **Sentiment Analysis**: Using VADER's lexicon-based approach to measure sentiment
- **Bag of Words**: Text representation focusing on word frequency rather than order

## Results

The analysis reveals several insights about the Amazon reviews:

- Distribution of sentiment across different star ratings
- Correlation between sentiment scores and user ratings
- Identification of common positive and negative sentiment triggers
- Entity recognition results highlighting frequently mentioned brands and products

## Skills Demonstrated

- **Natural Language Processing**: Applied tokenization, POS tagging, and sentiment analysis to extract meaning from unstructured text data
- **Data Analysis**: Used pandas for structured data manipulation and statistical analysis of sentiment patterns
- **Data Visualization**: Created clear, informative visualizations to communicate findings effectively
- **Python Programming**: Demonstrated proficiency in Python libraries for data science (NLTK, pandas, matplotlib)
- **Jupyter Notebook**: Used interactive computing for exploratory data analysis and result presentation
- **Machine Learning Concepts**: Applied pre-trained sentiment analysis models to classify text sentiment

## Future Improvements

- Implement custom sentiment analysis model trained specifically on e-commerce reviews
- Incorporate aspect-based sentiment analysis to identify sentiment toward specific product features
- Add topic modeling to group reviews by discussed themes
- Create interactive dashboard for exploring sentiment trends over time
- Expand analysis to include review helpfulness prediction

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


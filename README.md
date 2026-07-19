# Social Media Sentiment Analysis Dashboard

## Project Overview
This project implements a Machine Learning-powered sentiment analysis system.

## Features
- Text preprocessing
- TF-IDF feature extraction
- ML models
- Interactive dashboard
- Batch CSV sentiment prediction

CSV uploads accept `text`, `tweet`, `full_text`, `content`, `comment`,
`review`, `message`, or `caption` as the source text column.
CSV datasets created from [Xquik API](https://docs.xquik.com/api-reference/overview)
results or reviewed [TweetClaw](https://github.com/Xquik-dev/tweetclaw)
exports work when one of these text columns is present.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run sentiment_dashboard.py
```

Xquik is an independent third-party service. Not affiliated with X Corp.
"Twitter" and "X" are trademarks of X Corp.

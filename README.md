# Steam Reviews Sentiment with Bi-LSTM (Russian Language)

This repository contains a TensorFlow [implementation](https://github.com/nslyubaykin/steam_reviews_rus/blob/master/steam_reviews.ipynb) of sentiment model for video-games reviews, trained on the Russian segment of Steam.

# Data

Gamers' vocabulary may be very specific as some words may have very different meaning from what they have in a broad language (e.g bug does not mean insect but glitch). So for building successful video-games reviews sentiment analysis model closely related training corpus has to be used. For the purpose of this research the Russian segment of Steam was parsed. Each review is annotated with "Recommend" or "Not recommend" players feedback which was used as a sentiment y-label for training. Overall nearly 1.5 mln. reviews were collected, 84% are positive and 16% are negative. Data splitted by batches is inside [steam_sentiment_data](https://github.com/nslyubaykin/steam_reviews_rus/tree/master/steam_sentiment_data) folder.

# Pre-processing



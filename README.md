# Steam Reviews Sentiment with Bi-LSTM (Russian Language)

This repository contains a TensorFlow [implementation](https://github.com/nslyubaykin/steam_reviews_rus/blob/master/steam_reviews.ipynb) of sentiment model for video-games reviews, trained on the Russian segment of Steam.

# Data

Gamers' vocabulary may be specific as some words may have completely different meaning from what they have in a broad language (e.g bug does not mean insect but glitch). So for building successful video-games reviews sentiment analysis models, closely related training corpus has to be used. For the purpose of this research the Russian segment of Steam was parsed. Each review is annotated with "Recommend" or "Not recommend" player's feedback which was used as sentiment y-label for training. Overall nearly 1.5 mln. reviews were collected, 84% are positive and 16% are negative. Data splitted by batches is inside [steam_sentiment_data](https://github.com/nslyubaykin/steam_reviews_rus/tree/master/steam_sentiment_data) folder.

# Pre-processing

For the purpose of training models, too short or uniformative reviews were filtered out which left around 1.128 mln valid reviews. Raw texts were tokenized with [Youtokentome](https://github.com/VKCOM/YouTokenToMe) BPE tokenizer.

# Models

For text classification bidirectional LSTM many-to-one architecture is used. Before passing to LSTM layer tokens are embedded, after LSTM layer dimensionality is reduced with global max pooling.

# Quality metrics

On 50k test dataset model was able to achieve:

- Binary accuracy: 92.4%
- AUC ROC: 0.9519
- AUC PRC: 0.9891

# Minimal example:

After clonning this repository run from it:

Imports:
```python
import pandas as pd
from steam_utils.sentiment_utils import *
```
Load the models:
```python
# Loading model:
RSRNet = tf.keras.models.load_model('russian_steam_review_model/RSRNet.h5')
# Loading tokenizer:
steam_bpe = yttm.BPE(model='russian_steam_review_model/steam_tokenizer.model')
# Define scorer:
steam_scorer = ScoreReview(tokenizer=steam_bpe, score_net=RSRNet,
                           pad='post', truncate='post', truncate_len=400)
```

Your texts:
```python
sample_texts = ['Удалил спустя пару часов. Это просто невыносимо',
                'Эта игра настоящий шедевр! Она взорвала мои моооозгииии',
                'Одни баги, игра постоянно вылетает',
                'Захватывающий сюжет и интересный геймплей',
                'Отлично, если не играть',
                'Графон мыльный, куча доната и микротранзакций',
                'Динамичная боевка, наполненный открытый мир',
                'Зависает в главном меню',
                'Мой компьютер не потянет эту игру',
                'Вернул деньги',
                'Кидаю деньги в монитор']
```



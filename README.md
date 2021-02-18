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

Predict for your texts:
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
                
score, label, sentiment = steam_scorer.predict(sample_texts)
pd.DataFrame({'sample_review': sample_texts,
              'score': score,
              'label': label,
              'sentiment': sentiment})
```
"text/plain": [
       "                                        sample_review     score  label  \\\n",
       "0     Удалил спустя пару часов. Это просто невыносимо  0.311001      0   \n",
       "1   Эта игра настоящий шедевр! Она взорвала мои мо...  0.995141      1   \n",
       "2                  Одни баги, игра постоянно вылетает  0.061664      0   \n",
       "3           Захватывающий сюжет и интересный геймплей  0.997844      1   \n",
       "4                             Отлично, если не играть  0.925283      1   \n",
       "5       Графон мыльный, куча доната и микротранзакций  0.101197      0   \n",
       "6         Динамичная боевка, наполненный открытый мир  0.994680      1   \n",
       "7                             Зависает в главном меню  0.207798      0   \n",
       "8                   Мой компьютер не потянет эту игру  0.894053      1   \n",
       "9                                       Вернул деньги  0.258352      0   \n",
       "10                             Кидаю деньги в монитор  0.681771      1   \n",
       "11  В игре много минусов и их перечисление займет ...  0.045930      0   \n",
       "12  Для меня серия Ghost Recon закончилась на Tom ...  0.415476      0   \n",
       "13                                Не работают сервера  0.153678      0   \n",
       "14  1:54 меня этим и подкупила, что все смешали, м...  0.848872      1   \n",
       "15      Юбисофт опять обманули, ничего удивительного.  0.266738      0   \n",
       "16  Если честно не советую вам покупать, графика х...  0.356482      0   \n",
       "17  Я, уже 6 числа купил долгая игруля и норм игра...  0.889164      1   \n",
       "18  Игра понравилась. Тот случай, когда обзоры нев...  0.991546      1   \n",
       "19  Бета была многообещающей, но в финальную верси...  0.095586      0   \n",
       "20                     Зря потратил время на эту игру  0.156293      0   \n",
       "21  Судя по трейлерам, крутой блокбастер, но геймп...  0.379887      0   \n",
       "22                                Офигеть, какая игра  0.970906      1   \n",
       "23                                 Ну что это за игра  0.902225      1   \n",
       "24  Чудесная и великолепная графика, красивые зака...  0.309907      0   \n",
       "25  Видно в комментах люди уже не верят в игру. Он...  0.943102      1   \n",
       "26  Скорее бы уже вышла! После цусими не во что иг...  0.997520      1   \n",
       "27  Мда. Викинги никак не могут быть в списках асс...  0.194034      0   \n",
       "\n",
       "   sentiment  \n",
       "0   Negative  \n",
       "1   Positive  \n",
       "2   Negative  \n",
       "3   Positive  \n",
       "4   Positive  \n",
       "5   Negative  \n",
       "6   Positive  \n",
       "7   Negative  \n",
       "8   Positive  \n",
       "9   Negative  \n",
       "10  Positive  \n",
       "11  Negative  \n",
       "12  Negative  \n",
       "13  Negative  \n",
       "14  Positive  \n",
       "15  Negative  \n",
       "16  Negative  \n",
       "17  Positive  \n",
       "18  Positive  \n",
       "19  Negative  \n",
       "20  Negative  \n",
       "21  Negative  \n",
       "22  Positive  \n",
       "23  Positive  \n",
       "24  Negative  \n",
       "25  Positive  \n",
       "26  Positive  \n",
       "27  Negative  "
      ]


import tensorflow as tf
import youtokentome as yttm
import numpy as np

from math import ceil
from steam_utils.text_utils import *



def RSRNet_predict(model, newdata, batch_size=512):
    """
    Function for batched sentiment fit
    """
    iter_num = int(ceil(len(newdata) / batch_size))
    start = 0
    fit = np.array([float('nan')] * len(newdata))
    for itr in range(iter_num):
        end = min(start + batch_size, len(newdata))
        batch = newdata[start:end]
        fit[start:end] = model(batch).numpy().reshape(-1)
        start = end
    return fit


class ScoreReview():
    """
    Class for making predictions for raw texts
    with tokenizer and sentiment models
    """
    def __init__(self, tokenizer, score_net, pad, truncate, truncate_len):
        self.tokenizer = tokenizer
        self.score_net = score_net
        self.pttl = [pad, truncate, truncate_len]
        
    def predict(self, review_text_list):
        # Pre-processing:
        proc_texts = list(map(lambda text: clean_text(text.lower()), review_text_list))
        # Tokenizing
        encoded_texts = self.tokenizer.encode(proc_texts, output_type=yttm.OutputType.ID)
        # Padding & truncating sequences
        PAD, TRUNCATE, TRUNCATE_LEN = self.pttl
        X_new = tf.keras.preprocessing.sequence.pad_sequences(encoded_texts, maxlen=TRUNCATE_LEN, 
                                                              padding=PAD, truncating=TRUNCATE)
        # Obtaining fit from the net:
        raw_fit = RSRNet_predict(self.score_net, X_new)
        out_class = list((raw_fit > 0.5).astype('int32'))
        sentiment_decode = {1: 'Positive', 0: 'Negative'}
        out_sentiment = list(map(lambda cl: sentiment_decode[cl], out_class))
        return raw_fit, out_class, out_sentiment

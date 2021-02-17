import re
import string
import unicodedata


TO_REMOVE = ('обзор продукта в раннем доступе',
             'товар получен бесплатно',
             '',
             'средства за продукт возвращены')


def clean_review(review_text, to_remove=TO_REMOVE):
    """
    Function for removing not meaningful parts of game review
    """
    review_text = review_text.lower()
    review_vec = review_text.split('\n')[1:]
    review_vec = [part for part in review_vec if part not in to_remove]
    return '\n'.join(review_vec)


def filter_reviews(review_text):
    """
    This function containы several empirical conditions
    to detect unrelevant reviews
    """
    # condition on overall review length
    review_len = len(list(review_text))
    len_cond = review_len > 17 + 2
    # condition on number of words
    num_words = len(re.split(r'\s', review_text))
    word_cond = num_words >= 3
    # condition on russian characters share:
    rus_pattern = re.compile(r'[А-Яа-я]')
    rus_char_len = len(rus_pattern.findall(review_text))
    if review_len > 0:
        rus_cond = rus_char_len / review_len > 0.4
    else:
        rus_cond = False
    # aggregating all conditions
    cond_vec = [len_cond, word_cond, rus_cond]
    
    return all(cond_vec)


def remove_control_characters(s):
    """
    Finction for removing control characters
    """
    if s == s:
        out_string = "".join(ch if unicodedata.category(ch)[0] not in ["C", "Z"] else ' ' for ch in s)
        return re.sub(' +', ' ', out_string).strip()
    else:
        return float('nan')


def tokenize(review_text):
    """
    On Steam informal language is tokenized with ♥
    This function replaces it with token
    """
    swear_pattern = re.compile(r'([А-Яа-я])*([♥])+([А-Яа-я])*', re.IGNORECASE)
    return re.sub(swear_pattern, 'sweartoken', review_text)


def add_spaces(text):
    """
    Some sentences turned out to be glued together during
    parsing. That function fixes this nexative effect.
    """
    a = re.sub(r'(?<=[.!?:;,)])(?=[a-zа-я])', ' ', text)
    return a


def clean_text(review_text):
    """
    This function cleans reviews retrieved from Steam
    """
    # remove trash
    trash_pattern = re.compile(r'[^a-zа-яё\s{}😀-🙏🌀-🗿🚀-\U0001f6ff\U0001f1e0-🇿]'.format(re.escape(string.punctuation)))
    no_trash = re.sub(trash_pattern, '', tokenize(review_text))
    # remove numbers
    # needed to avoid overfitting to user impressions 
    # sometimes users rate the game inside the review
    # e.g 3 out of 10 
    no_num = re.sub(r'[0-9]+', '', no_trash)
    # remove repeating letters
    repeat_pattern = re.compile(r'(.)\1{2,}', re.IGNORECASE)
    no_rep = re.sub(repeat_pattern, r'\1', no_num)
    # adding missing spaces:
    no_rep = add_spaces(no_rep)
    # removing redundant spaces
    space_pattern = re.compile(r'\s{2,10}')
    no_space = re.sub(space_pattern, ' ', no_rep)
    # striping
    no_space = no_space.strip()
    out_text = no_space
    return out_text


def to_num_y(y_text):
    """
    Convert labels to numeric form
    """
    y_dict = {'Рекомендую': 1, 'Не рекомендую': 0}
    return y_dict[y_text]


def text_length(n_text):
    """
    Get text length
    """
    return len(n_text.split(' '))

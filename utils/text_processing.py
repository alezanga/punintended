import re
from typing import Union, Tuple, List, Set

import spacy


def get_lemma_word(text: str, nlp: spacy.Language, remove_stopwords: bool = False, join_result: bool = False) \
        -> Union[Tuple[List, List], Tuple[str, str]]:
    lemmatized_words = list()
    words = list()
    for w in nlp(text.lower()):
        if w.text.strip() and not w.is_punct and (not remove_stopwords or not w.is_stop):
            words.append(w.text.strip())
            lemmatized_words.append(w.lemma_)
    if join_result:
        lemmatized_words, words = " ".join(lemmatized_words).strip(), " ".join(words).strip()
    return lemmatized_words, words


def preprocessing(text: str) -> str:
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('utf-8')

    # Remove double quotes
    text = text.replace('"', '')

    # Strip leading and trailing whitespace
    return text.strip()


def get_lemmas_flat_set(_context_words_args, nlp: spacy.Language, remove_stopwords: bool = False,
                        return_lists: bool = False, join_composed_expressions: bool = False) \
        -> Union[Tuple[Set[str], Set[str]], Tuple[List[str], List[str]]]:
    """
    Lemmatize set/sequence of words

    :param _context_words_args: set or sequence of words to be lemmatized
    :param nlp: spacy model
    :param remove_stopwords: whether to remove stopwords
    :param return_lists: if True, return words and corresponding lemmas as two separate lists. If False return sets, meaning that eventual duplicate lemmas are removed.
    :return: tuple with sequences of words and corresponding lemmas
    """
    lemma_set = set() if not return_lists else list()
    word_set = set() if not return_lists else list()
    for ws in _context_words_args:
        lw, w = get_lemma_word(ws, nlp, remove_stopwords, join_composed_expressions)
        if not join_composed_expressions:
            for e in lw:
                lemma_set.add(e) if not return_lists else lemma_set.append(e)
            for e in w:
                word_set.add(e) if not return_lists else word_set.append(e)
        else:
            lemma_set.add(lw) if not return_lists else lemma_set.append(lw)
            word_set.add(w) if not return_lists else word_set.append(w)
    return lemma_set, word_set

import numpy as np
from numpy import ndarray


def generate_ngram(texts: list[str], ngram: list[int] = None) -> ndarray:
    if ngram is None:
        ngram = [1]
    phrase_dict: dict[str, int] = {}
    for text in texts:
        for span in ngram:
            for i in range(len(text) - span + 1):
                phrase = ' '.join(text.split()[i:i + span])
                if phrase not in phrase_dict:
                    phrase_dict[phrase] = len(phrase_dict)
    # A feature vector for every row
    features = np.zeros((len(texts), len(phrase_dict) + 1))
    for i, text in enumerate(texts):
        features[i][-1] = 1
        for span in ngram:
            for j in range(len(text) - span + 1):
                phrase = ' '.join(text.split()[j:j + span])
                features[i][phrase_dict[phrase]] += 1
    return features

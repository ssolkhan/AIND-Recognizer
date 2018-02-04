import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # implement the recognizer
    for i in range(test_set.num_items):
        single_word_probs = {}
        match_prob, match_word = float("-inf"), None
        
        seq, lengths = test_set.get_item_Xlengths(i)
        for word, model in models.items():
            try:
                single_word_probs[word] = model.score(seq, lengths)
            except Exception as e:
                single_word_probs[word] = float("-inf")
            
            if single_word_probs[word] > match_prob:
                match_prob, match_word = single_word_probs[word], word
                
        probabilities.append(single_word_probs)
        guesses.append(match_word)
        
    return probabilities, guesses

import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on BIC scores
        best_score, best_model  = float("inf"), None
        
        for num_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_components)
                logL = model.score(self.X, self.lengths)
                logN = np.log(self.X.shape[0])
                n_params = num_components * (num_components - 1) + 2 * self.X.shape[1] * num_components
                bic = -2 * logL + n_params * logN
                if bic < best_score:
                    best_score, best_model = bic, model
            except Exception as e:
                continue
    
        if best_model is None:
            return self.base_model(self.n_constant)
        else:
            return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Go through each model and compute its DIC score, only keeping track of the one with max DIC
        # If two DICs are maxs, return the model with lower complexity
        best_score, best_model = float("-inf"), None
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_components)
                logPXI = model.score(self.X, self.lengths)

                sumLogPXexceptI = 0.
                words = list(self.words.keys())
                M = len(words)
                words.remove(self.this_word)

                for word in words:
                    try:
                        modelSelExceptI = ModelSelector(self.words, self.hwords, word, self.n_constant, self.min_n_components, self.max_n_components, self.random_state, self.verbose)
                        sumLogPXexceptI += modelSelExceptI.base_model(num_components).score(modelSelExceptI.X, modelSelExceptI.lengths)
                    except:
                        M = M - 1

                dic  = logPXI - sumLogPXexceptI / (M - 1)

                if best_score is None or best_score < dic:
                  best_score, best_model = dic, model 
            except:
                pass

        if best_model is None:
            return self.base_model(self.n_constant)
        else:
            return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    num_folds = 3

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection using CV
        best_score, best_model = float("-inf"), None
        
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            scores, num_folds = [], SelectorCV.num_folds
            model, logL = None, None
            
            if(len(self.sequences) < num_folds):
                break
            
            #split_method = KFold(random_state=self.random_state, num_folds=self.num_folds)
            split_method = KFold(SelectorCV.num_folds)
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                X_test,  lengths_test  = combine_sequences(cv_test_idx, self.sequences)
                try:
                    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                    random_state=inst.random_state, verbose=False).fit(X_train, lengths_train)
                    logL = model.score(X_test, lengths_test)
                    scores.append(logL)
                except Exception as e:
                    break
            
            avg = float("-inf")
            if len(scores) > 0:
                avg = np.average(scores)
            
            if avg > best_score:
                best_score, best_model = avg, model
        
        if best_model is None:
            return self.base_model(self.n_constant)
        else:
            return best_model

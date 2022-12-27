import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer


class naive_bayes_spam():

    def __init__(self, data, alpha=1, language='english'):
        self.alpha = alpha
        self.language = language
        self.data = data


    def clean(self):
        self.data['MESSAGE'] = self.data['MESSAGE'].str.lower()
        self.data['MESSAGE'] = self.data['MESSAGE'].str.replace('[^a-z ]', '')


    def set_vocabulary(self):
        count_vector = CountVectorizer(stop_words=self.language)
        freq_table = count_vector.fit_transform(self.data['MESSAGE']).toarray()
        self.vocabulary = count_vector.get_feature_names_out()
        self.freq_table = pd.DataFrame(freq_table, columns=self.vocabulary)

        self.data = pd.concat([self.data, self.freq_table], axis=1)


    def set_prior(self):
        self.spam = self.data[self.data['TYPE'] == 1]
        self.ham = self.data[self.data['TYPE'] == 0]

        self.p_spam = len(self.spam) / len(self.data)
        self.p_ham = len(self.ham) / len(self.data)

        n_words_per_spam = self.spam['MESSAGE'].apply(
            lambda x: len(x.split(' '))).sum()
        n_words_per_ham = self.ham['MESSAGE'].apply(
            lambda x: len(x.split(' '))).sum()

        self.n_spam = n_words_per_spam.sum()
        self.n_ham = n_words_per_ham.sum()

        self.n_vocabulary = len(self.vocabulary)
        self.spam_params = {unique_word: 0 for unique_word in self.vocabulary}
        self.ham_params = {unique_word: 0 for unique_word in self.vocabulary}


    def fit(self):
        self.clean()
        self.set_vocabulary()
        self.set_prior()

        for word in self.vocabulary:
            self.n_word_given_spam = self.spam[word].sum()
            self.p_word_given_spam = (
                self.n_word_given_spam + self.alpha) / (self.n_spam + self.alpha * self.n_vocabulary)
            self.spam_params[word] = self.p_word_given_spam

            self.n_word_given_ham = self.ham[word].sum()
            self.p_word_given_ham = (
                self.n_word_given_ham + self.alpha) / (self.n_ham + self.alpha * self.n_vocabulary)
            self.ham_params[word] = self.p_word_given_ham


    def predict(self, messages):

        y_pred = []
        for message in messages:
            message = message.lower()
            message = message.replace('[^a-z ]', '')
            message = message.split(' ')

            p_spam_given_message = self.p_spam
            p_ham_given_message = self.p_ham

            for word in message:
                if word in self.spam_params:
                    p_spam_given_message *= self.spam_params[word]

                if word in self.ham_params:
                    p_ham_given_message *= self.ham_params[word]

            if p_ham_given_message > p_spam_given_message:
                y_pred.append(0)
            elif p_ham_given_message < p_spam_given_message:
                y_pred.append(1)
            else:
                y_pred.append(0)

        return y_pred

    def save_model(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

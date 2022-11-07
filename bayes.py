import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, recall_score
import warnings
warnings.filterwarnings("ignore")


def create_data(X, y):

    train_data = pd.concat([X, y], axis=1, ignore_index=True)
    train_data.columns = ['MESSAGE', 'TYPE']
    train_data.reset_index(drop=True, inplace=True)

    return train_data


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

        n_words_per_spam = self.spam['MESSAGE'].apply(lambda x: len(x.split(' '))).sum()
        n_words_per_ham = self.ham['MESSAGE'].apply(lambda x: len(x.split(' '))).sum()

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
            self.p_word_given_spam = (self.n_word_given_spam + self.alpha) / (self.n_spam + self.alpha * self.n_vocabulary)
            self.spam_params[word] = self.p_word_given_spam

            self.n_word_given_ham = self.ham[word].sum()
            self.p_word_given_ham = (self.n_word_given_ham + self.alpha) / (self.n_ham + self.alpha * self.n_vocabulary)
            self.ham_params[word] = self.p_word_given_ham

    def predict(self, message):
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

        return p_spam_given_message, p_ham_given_message


@click.command()
@click.option('--alpha', default=0.5, help='Laplace smoothing parameter')
@click.option('--language', default='english', help='Language for stop words')
@click.option('--mode', default='demo', help='Mode to use')
@click.option('--message', default='Hello, how are you?', help='Message to predict')
def main(alpha: float, language: str, mode: str, message: str=None):
    """Naive Bayes model implementation for spam detection
    - In order to train this model, a csv file with two columns is required,
    there must be a column with the messages called "MESSAGE" and another with the labels
    called "TYPE", where "spam" is for spam and "ham" is for messages that are not spam.
    """

    def train_model():
        print('-' * 50)
        print('Running demo mode on given training data')
        df = pd.read_csv('data/sms_spam.csv', encoding='latin-1', names=['TYPE', 'MESSAGE'], skiprows=1)
        df['TYPE'] = df['TYPE'].map({'ham': 0, 'spam': 1})

        X_train, X_test, y_train, y_test = train_test_split(df['MESSAGE'], df['TYPE'], random_state=42)

        print('Training model...')
        train_data = create_data(X_train, y_train)
        test_data = create_data(X_test, y_test)
        clf = naive_bayes_spam(train_data, alpha=alpha, language=language)
        clf.fit()

        y_pred = []
        for message in test_data['MESSAGE']:
            p_spam, p_ham = clf.predict(message)
            if p_ham > p_spam:
                y_pred.append(0)
            elif p_ham < p_spam:
                y_pred.append(1)
            else:
                y_pred.append(0)

        print('Model metrics on Test Data:')
        print('Accuracy:', accuracy_score(test_data['TYPE'], y_pred))
        print('Recall:', recall_score(test_data['TYPE'], y_pred))
        print('-' * 50)

        return clf

    if mode == 'demo':

        _ = train_model()

    elif mode == 'predict':

        clf = train_model()
        print('Training model...')
        print('Running prediction mode on given message')
        print('Message:', message)
        print('Predicting...')
        p_spam, p_ham = clf.predict(message)

        print('-' * 50)
        print('Spam probability:', p_spam)
        print('Ham probability:', p_ham)
        print('-' * 50)

        print('Message is spam:', p_spam > p_ham)


if __name__ == '__main__':
    main()
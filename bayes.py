import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from naive_bayes import naive_bayes_spam
import warnings
warnings.filterwarnings("ignore")

# Helper function
def create_data(X, y):

    train_data = pd.concat([X, y], axis=1, ignore_index=True)
    train_data.columns = ['MESSAGE', 'TYPE']
    train_data.reset_index(drop=True, inplace=True)

    return train_data


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
        y_pred = clf.predict(test_data['MESSAGE'])
        
        print(f'''
        Model metrics on Test Data:
        Accuracy: {accuracy_score(test_data['TYPE'], y_pred)}
        Recall: {recall_score(test_data['TYPE'], y_pred)}
        {'-' * 50}
        ''')

        clf.save_model('model.pkl')


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
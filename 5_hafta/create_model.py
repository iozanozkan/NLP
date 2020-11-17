import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import nltk
import re


fake=pd.read_csv('datasets/Fake.csv')
real=pd.read_csv('datasets/True.csv')

fake["label"]=0
df=pd.concat([fake.text, fake.label], axis=1)

real["label"]=1
df2=pd.concat([real.text, real.label], axis=1)

result = pd.concat([df, df2])

stop_word_list = nltk.corpus.stopwords.words('english')

docs = result.text
docs = docs.map(lambda x: re.sub(r"[-()\"#/@;:<>{}+=~|.!?,]", '', x))
docs = docs.map(lambda x: x.lower())
docs = docs.map(lambda x: x.strip())

def token(values):
    filtered_words = [word for word in values.split() if word not in stop_word_list]
    not_stopword_doc = " ".join(filtered_words)
    return not_stopword_doc


docs = docs.map(lambda x: token(x))
result.text = docs

X = result.text
y = result.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 100)

text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LogisticRegression()),
                      ])

text_clf.fit(X_train, y_train)
text_clf.score(X_test, y_test)

y_pred = text_clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


joblib.dump(text_clf, "model.pkl")
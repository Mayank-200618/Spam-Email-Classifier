# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Dataset
data = {
    'message': [
        'Win money now',
        'Hello how are you',
        'Claim your prize',
        'Let’s meet tomorrow',
        'Free entry in contest',
        'Are you available?',
        'Win cash prize',
        'Call me now'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])

y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test prediction
sample = ["Win a free ticket now"]
sample_vec = vectorizer.transform(sample)

prediction = model.predict(sample_vec)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam Message")

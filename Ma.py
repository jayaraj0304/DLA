import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("data/quora_sample.csv")

q1 = df["question1"].fillna("")
q2 = df["question2"].fillna("")
y = df["is_duplicate"]

# 2. Combine questions for TF-IDF
questions = pd.concat([q1, q2])

tfidf = TfidfVectorizer()
tfidf.fit(questions)

# 3. Convert questions into vectors
v1 = tfidf.transform(q1)
v2 = tfidf.transform(q2)

# 4. Extract pairwise features
from scipy.sparse import hstack

X = hstack([abs(v1 - v2), v1.multiply(v2)])

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 6. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Predict and evaluate
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Predictions:", pred)

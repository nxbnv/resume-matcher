import pickle

# Try loading locally to see if the file itself is corrupted
with open("models/vectorizer.pkl", "rb") as f:
    obj = pickle.load(f)
    print("Vectorizer loaded successfully!")

with open("models/bert_model.pkl", "rb") as f:
    obj = pickle.load(f)
    print("BERT Model loaded successfully!")
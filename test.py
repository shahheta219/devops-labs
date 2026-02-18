import pickle
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

accuracy = model.score(X, y)
print("Accuracy:", accuracy)

if accuracy < 0.8:
    raise Exception("Model accuracy too low!")

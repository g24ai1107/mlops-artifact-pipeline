import joblib
from sklearn.datasets import load_digits

def main():
    model = joblib.load("model_train.pkl")
    digits = load_digits()
    X = digits.data
    predictions = model.predict(X)
    print("Sample predictions:", predictions[:10])

if __name__ == "__main__":
    main()


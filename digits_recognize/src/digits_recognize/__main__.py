import pickle
from matplotlib import pyplot
from pathlib import Path


def main() -> None:
    DATA_PATH = Path(__file__).parent / "data"
    PATH = DATA_PATH / "mnist.pkl"
    with PATH.open("rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
        print(f"Training set: {x_train.shape}, {y_train.shape}")
        print(f"Validation set: {x_valid.shape}, {y_valid.shape}")

    pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
    pyplot.show()


if __name__ == "__main__":
    main()

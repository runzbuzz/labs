import argparse
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Input


def load_dataset(dataset_name):
    if dataset_name == "mnist":
        mnist = keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        return (x_train, y_train), (x_test, y_test)
    else:
        raise ValueError("Invalid dataset")


def create_model(pool, activation):
    """
    This is inspired by the le_net_5 for digit recognition
    :param pool:
    :param activation:
    :return:
    """
    keras.utils.set_random_seed(7)
    if pool == "avg":
        pool_func = keras.layers.AveragePooling2D
    elif pool == "max":
        pool_func = keras.layers.MaxPool2D
    else:
        raise ValueError(f"Unknown pool function: {pool}")

    model = Sequential(
        [Input(shape=(28, 28, 1)), Conv2D(filters=6, kernel_size=5, strides=(1, 1), activation=activation),
         pool_func(pool_size=(2, 2), strides=(2, 2)),
         Conv2D(filters=16, kernel_size=5, strides=(1, 1), activation=activation),
         pool_func(pool_size=(2, 2), strides=(2, 2)),

         keras.layers.Flatten(), keras.layers.Dense(120, activation=activation),
         keras.layers.Dense(84, activation=activation), keras.layers.Dense(10, activation="softmax")])
    return model


def get_model_path(pool, activation, epochs, batch_size):
    fname = f"{pool}_{activation}_{batch_size}_{epochs}.keras"
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{fname}"
    return model_path


def get_results_path(pool, activation, epochs, batch_size):
    fname = f"{pool}_{activation}_{batch_size}_{epochs}_results.json"
    os.makedirs("results", exist_ok=True)
    model_path = f"results/{fname}"
    return model_path


def load_model(pool, activation, epochs, batch_size):
    model_path = get_model_path(pool, activation, epochs, batch_size)
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
    return None


def get_history_path(pool, activation, epochs, batch_size):
    model_path = get_model_path(pool, activation, epochs, batch_size)
    history_path = model_path.replace(".keras", "_history.json")
    return history_path


def load_history(pool, activation, epochs, batch_size):
    history_path = get_history_path(pool, activation, epochs, batch_size)
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        return history
    return None


def save_history(history_dict, pool, activation, epochs, batch_size):
    history_path = get_history_path(pool, activation, epochs, batch_size)
    with open(history_path, "w") as f:
        json.dump(history_dict, f)


def get_model(x_train, y_train, pool, activation, epochs, batch_size):
    model = load_model(pool, activation, epochs, batch_size)
    history = load_history(pool, activation, epochs, batch_size)
    if model and history:
        return model, history

    model = create_model(pool, activation)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_obj = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    model_path = get_model_path(pool, activation, epochs=epochs, batch_size=batch_size)
    model.save(model_path)
    save_history(history_obj.history, pool, activation, epochs, batch_size)
    history = load_history(pool, activation, epochs, batch_size)
    return model, history


def workflow(pool, activation, epochs, batch_size):
    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    model, history = get_model(x_train, y_train, pool, activation, epochs=epochs, batch_size=batch_size)
    loss, accuracy = model.evaluate(x_test, y_test)
    save_results(pool=pool, activation=activation, epochs=epochs, batch_size=batch_size, loss=loss, accuracy=accuracy)
    return history, loss, accuracy


def get_pool_title(pool):
    if pool == "avg":
        return "Average Pooling"
    elif pool == "max":
        return "Max Pooling"
    raise ValueError(f"Unknown pooling method: {pool}")


def get_activation_title(activation):
    if activation == "relu":
        return "ReLU"
    elif activation == "tanh":
        return "Tanh"
    elif activation == "sigmoid":
        return "Sigmoid"
    elif activation == "swish":
        return "Swish"
    elif activation == "gelu":
        return "GELU"
    elif activation in ["leaky_relu", "leaky-relu"]:
        return "Leaky ReLU"
    elif activation == "elu":
        return "ELU"
    raise ValueError(f"Unknown activation function: {activation}")


def plot_history(history, pool, activation):
    df = pd.DataFrame(history)
    df_acc = df[['accuracy', 'val_accuracy']]
    df_acc.rename(columns={'accuracy': 'train', 'val_accuracy': 'validation'}, inplace=True)
    df_acc.plot(title=f'Model Accuracy: {get_pool_title(pool)} + {get_activation_title(activation)}',
                figsize=(12, 8)).set(xlabel='Epoch', ylabel='Accuracy')
    plt.show()


def save_results(pool, activation, epochs, batch_size, loss, accuracy):
    results_path = get_results_path(pool, activation, epochs, batch_size)

    results = {"test_accuracy": float(accuracy), "test_loss": float(loss)}

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


def generate_charts():
    datasets = []

    for fname in os.listdir("models"):
        if not fname.endswith("_history.json"):
            continue

        path = os.path.join("models", fname)

        with open(path, "r") as f:
            history = json.load(f)

        # Example: avg_relu_128_10_history.json
        pool, activation, batch_size, epochs = (
            fname.removesuffix("_history.json").replace("leaky_relu", "leaky-relu").split("_"))

        datasets.append(
            {"label": f"{get_pool_title(pool)} + {get_activation_title(activation)}", "data": history["val_accuracy"],
             "fill": False, "tension": 0.2})

    if not datasets:
        raise ValueError("No training history files found.")

    chart_data = {"labels": list(range(1, max(len(d["data"]) for d in datasets) + 1)), "datasets": datasets}

    with open("results/charts_js_data.js", "w") as f:
        json.dump(chart_data, f, indent=4)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="LeNet-5-inspired CNN for MNIST classification")

    parser.add_argument("--pool", type=str, choices=["avg", "max"], default="avg", help="Pooling method (default: avg)")

    parser.add_argument("--activation", type=str,
                        choices=["relu", "tanh", "sigmoid", "swish", "gelu", "leaky_relu", "elu"], default="relu",
                        help="Hidden-layer activation function (default: relu)")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")

    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size (default: 128)")

    parser.add_argument("--charts", action="store_true", help="Generate Chart.js comparison data")

    args = parser.parse_args()

    if args.charts:
        generate_charts()
        return

    if args.epochs <= 0:
        parser.error("--epochs must be a positive integer")

    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")

    history, loss, accuracy = workflow(pool=args.pool, activation=args.activation, epochs=args.epochs,
                                       batch_size=args.batch_size)

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.2%}")
    plot_history(history, pool=args.pool, activation=args.activation)


if __name__ == "__main__":
    main()

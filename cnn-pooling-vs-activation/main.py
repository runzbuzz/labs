import argparse
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow_datasets as tfds
from itertools import product
from statistics import mean
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Input

ACTIVATION_TITLES = {"relu": "ReLU", "tanh": "Tanh", "sigmoid": "Sigmoid", "swish": "Swish", "gelu": "GELU",
                     "leaky_relu": "Leaky ReLU", "elu": "ELU"}

POOL_TITLES = {"avg": "Average Pooling", "max": "Max Pooling"}


def get_tf_dataset_name(dataset_name):
    datasets = {"emnist-balanced": "emnist/balanced"}
    return datasets.get(dataset_name, dataset_name)


def load_dataset(dataset_name):
    tf_dataset_name = get_tf_dataset_name(dataset_name)
    dataset_train, dataset_test = tfds.load(tf_dataset_name, split=["train", "test"], batch_size=-1, as_supervised=True)
    (x_train, y_train) = tfds.as_numpy(dataset_train)
    (x_test, y_test) = tfds.as_numpy(dataset_test)

    # reduce the memory
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    x_train /= 255.0
    x_test /= 255.0

    return (x_train, y_train), (x_test, y_test)


def parse_fname(fname, suffix):
    fetched_dataset_name, pool, activation, batch_size, epochs, seed = (fname.removesuffix(suffix).split("__"))

    return fetched_dataset_name, pool, activation, batch_size, epochs, seed


def create_model(pool, activation, num_classes):
    """
    This is inspired by the le_net_5 for digit recognition
    :param pool:
    :param activation:
    :param num_classes: MNIST has 10 but others might have more.
    :return:
    """
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
         keras.layers.Dense(84, activation=activation), keras.layers.Dense(num_classes, activation="softmax")])
    return model


def get_model_path(dataset_name, pool, activation, epochs, batch_size, seed):
    fname = f"{dataset_name}__{pool}__{activation}__{batch_size}__{epochs}__{seed}.keras"
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", fname)
    return model_path


def get_results_path(dataset_name, pool, activation, epochs, batch_size, seed):
    fname = f"{dataset_name}__{pool}__{activation}__{batch_size}__{epochs}__{seed}_results.json"
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", fname)
    return results_path


def load_model(dataset_name, pool, activation, epochs, batch_size, seed):
    model_path = get_model_path(dataset_name, pool, activation, epochs, batch_size, seed=seed)
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
    return None


def get_history_path(dataset_name, pool, activation, epochs, batch_size, seed):
    model_path = get_model_path(dataset_name, pool, activation, epochs, batch_size, seed=seed)
    history_path = model_path.replace(".keras", "_history.json")
    return history_path


def load_history(dataset_name, pool, activation, epochs, batch_size, seed):
    history_path = get_history_path(dataset_name, pool, activation, epochs, batch_size, seed=seed)
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        return history
    return None


def save_history(history_dict, dataset_name, pool, activation, epochs, batch_size, seed):
    history_path = get_history_path(dataset_name, pool, activation, epochs, batch_size, seed)
    with open(history_path, "w") as f:
        json.dump(history_dict, f)


def get_num_classes(dataset_name):
    tf_dataset_name = get_tf_dataset_name(dataset_name)

    info = tfds.builder(tf_dataset_name).info
    num_classes = info.features["label"].num_classes
    return num_classes


def get_model(dataset_name, x_train, y_train, pool, activation, epochs, batch_size, seed, verbose=1):
    model = load_model(dataset_name, pool, activation, epochs, batch_size, seed=seed)
    history = load_history(dataset_name, pool, activation, epochs, batch_size, seed=seed)
    if model and history:
        return model, history

    model = create_model(pool, activation, num_classes=get_num_classes(dataset_name))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_obj = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                            verbose=verbose)
    model_path = get_model_path(dataset_name, pool, activation, epochs=epochs, batch_size=batch_size, seed=seed)
    model.save(model_path)
    save_history(history_obj.history, dataset_name, pool, activation, epochs, batch_size, seed=seed)
    history = load_history(dataset_name, pool, activation, epochs, batch_size, seed=seed)
    return model, history


def get_activation_title(activation):
    if activation not in ACTIVATION_TITLES:
        raise ValueError(f"Unknown activation function: {activation}")

    return ACTIVATION_TITLES[activation]


def get_pool_title(pool):
    if pool not in POOL_TITLES:
        raise ValueError(f"Unknown pooling method: {pool}")

    return POOL_TITLES[pool]


def plot_history(history, pool, activation):
    df = pd.DataFrame(history)
    df_acc = df[['accuracy', 'val_accuracy']]
    df_acc.rename(columns={'accuracy': 'train', 'val_accuracy': 'validation'}, inplace=True)
    df_acc.plot(title=f'Model Accuracy: {get_pool_title(pool)} + {get_activation_title(activation)}',
                figsize=(12, 8)).set(xlabel='Epoch', ylabel='Accuracy')
    plt.show()


def save_results(dataset_name, pool, activation, epochs, batch_size, loss, accuracy, seed):
    results_path = get_results_path(dataset_name, pool, activation, epochs, batch_size, seed=seed)

    results = {"test_accuracy": float(accuracy), "test_loss": float(loss)}

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


def generate_validation_charts(dataset_name):
    datasets = []

    for fname in os.listdir("models"):
        if not fname.endswith("_history.json"):
            continue

        path = os.path.join("models", fname)

        with open(path, "r") as f:
            history = json.load(f)

        # Example: avg_relu_128_10_history.json
        fetched_dataset_name, pool, activation, batch_size, epochs, seed = parse_fname(fname=fname,
                                                                                       suffix="_history.json")

        if dataset_name != fetched_dataset_name:
            continue

        datasets.append(
            {"label": f"{get_pool_title(pool)} + {get_activation_title(activation)}", "data": history["val_accuracy"],
             "fill": False, "tension": 0.2})

    if not datasets:
        raise ValueError("No training history files found.")

    chart_data = {"labels": list(range(1, max(len(d["data"]) for d in datasets) + 1)), "datasets": datasets}

    with open(f"results/{dataset_name}_charts_js_validation_data.js", "w") as f:
        json.dump(chart_data, f, indent=4)
        f.write("\n")


def generate_testing_charts(dataset_name):
    labels = []
    accuracies = []

    for fname in sorted(os.listdir("results")):
        if not fname.endswith("_results.json"):
            continue

        with open(os.path.join("results", fname), "r") as f:
            results = json.load(f)

        fetched_dataset_name, pool, activation, _, _, _ = parse_fname(fname=fname, suffix="_results.json")

        if dataset_name != fetched_dataset_name:
            continue

        labels.append(f"{get_pool_title(pool)} + {get_activation_title(activation)}")

        accuracies.append(results["test_accuracy"])

    if not labels:
        raise ValueError("No testing results found.")

    chart_data = {"labels": labels, "datasets": [{"label": "Test Accuracy", "data": accuracies}]}

    with open(f"results/{dataset_name}_charts_js_testing_data.js", "w") as f:
        json.dump(chart_data, f, indent=4)
        f.write("\n")


def generate_charts(dataset_name):
    generate_validation_charts(dataset_name)
    generate_testing_charts(dataset_name)


def workflow(dataset_name, pool, activation, epochs, batch_size, verbose, seed, dataset=None):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    if dataset is None:
        dataset = load_dataset(dataset_name)

    (x_train, y_train), (x_test, y_test) = dataset
    model, history = get_model(dataset_name, x_train, y_train, pool, activation, epochs=epochs, batch_size=batch_size,
                               verbose=verbose, seed=seed)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=verbose)
    save_results(dataset_name, pool=pool, activation=activation, epochs=epochs, batch_size=batch_size, loss=loss,
                 accuracy=accuracy, seed=seed)
    return history, loss, accuracy


def get_agg_files(dataset_name, epochs, batch_sizes, activations, pools, path, suffix):
    fnames = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(suffix):
            continue

        fetched_dataset_name, pool, activation, batch_size, epoch, _ = parse_fname(fname, suffix=suffix)

        if fetched_dataset_name != dataset_name:
            continue

        if pools and pool not in pools:
            continue

        if epochs and int(epoch) not in epochs:
            continue

        if activations and activation not in activations:
            continue

        if batch_sizes and int(batch_size) not in batch_sizes:
            continue

        fnames.append(fname)

    return fnames


def agg_results(dataset_name, epochs, batch_sizes, activations, pools, overwrite, verbose):
    """
    Aggregate testing results from matching experimental runs.

    Calculates the mean, maximum, minimum, and number of
    observations for each metric across all matching seeds.

    :param dataset_name: Dataset name.
    :param epochs: List of epoch counts to include.
    :param batch_sizes: List of batch sizes to include.
    :param activations: List of activation functions to include.
    :param pools: List of pooling methods to include.
    :param overwrite: "a" to ask, "y" to overwrite,
                      or "n" to skip existing files.
    :param verbose: Aggregation verbosity:
                0=silent,
                1=warnings and summaries,
                2=detailed file-level logs.
    :return: None.

    Empty filter lists include all available values.
    """
    pool_name = "all"
    activation_name = "all"
    batch_name = "all"
    epoch_name = "all"
    if len(pools) == 1:
        pool_name = pools[0]
    if len(activations) == 1:
        activation_name = activations[0]
    if len(batch_sizes) == 1:
        batch_name = str(batch_sizes[0])
    if len(epochs) == 1:
        epoch_name = str(epochs[0])

    out_fname = f"{dataset_name}__{pool_name}__{activation_name}__{batch_name}__{epoch_name}__agg-results.json"
    out_path = os.path.join("results", out_fname)
    if os.path.exists(out_path):
        overwrite = overwrite.strip().lower()
        if overwrite not in ("a", "y", "n"):
            raise ValueError(f"Invalid overwrite option: {overwrite}")

        if overwrite == "a":
            answer = input(f"Aggregated results already exist in {out_path}. "
                           "Overwrite? [y/N]: ")
            answer = answer.strip().lower()
        else:
            answer = overwrite
        if answer != "y":
            return None

    fnames = get_agg_files(dataset_name, epochs=epochs, batch_sizes=batch_sizes, activations=activations, pools=pools,
                           path="results", suffix="_results.json")

    if not fnames:
        if verbose > 0:
            print(f"Warning: No matching result files found for "
                  f"dataset={dataset_name}, pool={pool_name}, "
                  f"activation={activation_name}, batch_size={batch_name}, "
                  f"epochs={epoch_name}. Skipping aggregation.")
        return

    j_list = {}
    for fname in fnames:
        if verbose > 1:
            print(f"aggregating file: {fname}")
        with open(os.path.join("results", fname)) as f:
            j = json.load(f)
        for k, v in j.items():
            j_list.setdefault(k, []).append(v)

    j_agg = {}
    for k, values in j_list.items():
        j_agg[k] = {"mean": mean(values), "max": max(values), "min": min(values), "num_runs": len(values)}

    with open(out_path, "w") as f:
        json.dump(j_agg, f, indent=4)

    if verbose > 0:
        print(f"Successfully aggregated {len(fnames)} results into {out_fname}.")


def aggregate_results_per_activation_pool(dataset_name, epochs, batch_sizes, activations, pools, verbose, overwrite):
    # Aggregate the results for different seeds
    for activation in activations:
        for pool in pools:
            agg_results(dataset_name, epochs=epochs, batch_sizes=batch_sizes, activations=[activation], pools=[pool],
                        verbose=verbose, overwrite=overwrite)


def aggregate_results_per_pool(dataset_name, epochs, batch_sizes, activations, pools, verbose, overwrite):
    for pool in pools:
        agg_results(dataset_name, epochs=epochs, batch_sizes=batch_sizes, activations=activations, pools=[pool],
                    verbose=verbose, overwrite=overwrite)


def aggregate_results_per_activation(dataset_name, epochs, batch_sizes, activations, pools, verbose, overwrite):
    for activation in activations:
        agg_results(dataset_name, epochs=epochs, batch_sizes=batch_sizes, activations=[activation], pools=pools,
                    verbose=verbose, overwrite=overwrite)


def aggregate(dataset_name, epochs, batch_sizes, activations, pools, verbose, overwrite):
    aggregate_results_per_activation_pool(dataset_name, epochs, batch_sizes, activations, pools, verbose=verbose,
                                          overwrite=overwrite)
    aggregate_results_per_activation(dataset_name, epochs, batch_sizes, activations, pools, verbose=verbose,
                                     overwrite=overwrite)
    aggregate_results_per_pool(dataset_name, epochs, batch_sizes, activations, pools, verbose=verbose,
                               overwrite=overwrite)


def main():
    pooling_methods = list(POOL_TITLES)
    activation_functions = list(ACTIVATION_TITLES)

    parser = argparse.ArgumentParser(description="LeNet-5-inspired CNN for MNIST classification")
    parser.add_argument("--pool", choices=pooling_methods + ["all"], default="all",
                        help="Pooling method (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, ], help="Random seeds for repeated experiments.")
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashion_mnist", "kmnist", "emnist-balanced"],
                        required=True, help="Dataset name")
    parser.add_argument("--activation", choices=activation_functions + ["all"], default="all",
                        help="Hidden-layer activation function (default: all)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size (default: 128)")
    parser.add_argument("--charts", action="store_true", help="Generate Chart.js comparison data")
    parser.add_argument("--plot", action="store_true",
                        help="Display training and validation accuracy plots (default: disabled)")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0,
                        help="Training verbosity: 0=silent, 1=progress bar, 2=epoch logs (default: 0)")
    parser.add_argument("--agg", action="store_true", help="Aggregate experimental results across all pooling "
                                                           "methods and activation functions for the selected "
                                                           "dataset, epoch count, and batch size.")
    parser.add_argument("--overwrite", type=str, default="a", choices=["a", "y", "n"],
                        help="Only for the aggregation, a for ask, y for yes means to overwrite, and n for not overwriting")
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be a positive integer")

    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")

    dataset_name = args.dataset

    pools = pooling_methods if args.pool == "all" else [args.pool]
    activations = (activation_functions if args.activation == "all" else [args.activation])

    if args.agg:
        aggregate(dataset_name, epochs=[args.epochs], batch_sizes=[args.batch_size], activations=activation_functions,
                  pools=pooling_methods, verbose=args.verbose, overwrite=args.overwrite)

    if args.charts:
        generate_charts(dataset_name)

    if args.charts or args.agg:
        return

    dataset = load_dataset(dataset_name)

    for pool, activation in product(pools, activations):
        for seed in args.seeds:
            print(f"\nRunning: {args.dataset} | {pool} | {activation} | {seed}")

            history, loss, accuracy = workflow(dataset_name=dataset_name, pool=pool, activation=activation,
                                               epochs=args.epochs, batch_size=args.batch_size, verbose=args.verbose,
                                               seed=seed, dataset=dataset)

            print(f"Test loss: {loss:.4f}")
            print(f"Test accuracy: {accuracy:.2%}")

            if args.plot:
                plot_history(history, pool=pool, activation=activation)


if __name__ == "__main__":
    main()

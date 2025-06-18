import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
parser.add_argument("--min_samples_split", type=int, default=2)
parser.add_argument("--min_samples_leaf", type=int, default=1)
parser.add_argument("--dataset", type=str, default="preprocessed_dataset")
args = parser.parse_args()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    X_train = pd.read_csv(f"{args.dataset}/X_train.csv")
    X_test = pd.read_csv(f"{args.dataset}/X_test.csv")
    Y_train = pd.read_csv(f"{args.dataset}/Y_train.csv").values.ravel()
    Y_test = pd.read_csv(f"{args.dataset}/Y_test.csv").values.ravel()

    with mlflow.start_run():
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)

        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42,
        )

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Y prediction
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
                            
        # Metrik training
        train_mse = mean_squared_error(Y_train, Y_train_pred)
        train_mae = mean_absolute_error(Y_train, Y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(Y_train, Y_train_pred)

        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2_score", train_r2)

        # Metrik testing
        test_mse = mean_squared_error(Y_test, Y_test_pred)
        test_mae = mean_absolute_error(Y_test, Y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(Y_test, Y_test_pred)

        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2_score", test_r2)
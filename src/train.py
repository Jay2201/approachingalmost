import joblib
import pandas as pd
from pandas.io import parsers
from sklearn import tree
from sklearn import metrics
import config
import os
import argparse

def run(fold):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # training data is where kfold is not equal to folds provided
    # note that we are resetting the index also
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where the kfolds is equal to folds provided
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label & convert training data to numpy array for both training & validation data
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values

    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    # initialize simple Decision Tree Classifier
    clf = tree.DecisionTreeClassifier()

    # fit the model on the training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # calculate & print the accuracy score of the model
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    # save the model
    joblib.dump(
            clf,
            os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add different argument you need & their type
    parser.add_argument(
        "--fold",
        type=int
    )

    # read the argument from command line
    args = parser.parse_args()

    # run the folds specified by command line
    run(fold=args.fold)
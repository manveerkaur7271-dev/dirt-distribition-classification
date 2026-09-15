#!/usr/bin/env python3
# Homework 3 classification demonstration script
#
# Uses the DirtDistribution classes from the Vacuum World
# Demonstrates the different behaviors of some mainline classifiers
#

# Standard library imports
import argparse

# Non-standard library dependencies
# import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder


# ** **********************************************************
# Model loaders
# ** **********************************************************


def load_svm():
    # Support Vector Machine - should be very best model here
    # https://en.wikipedia.org/wiki/Support_vector_machine
    from sklearn.svm import SVC

    return SVC(kernel="rbf")


def load_decision_tree(depth: int):
    # Arbitrarily deep decision tree - can overfit
    # https://en.wikipedia.org/wiki/Decision_tree
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(max_depth=depth, random_state=42)


def load_linear():
    # "Linear classifier" in sklearn terms = Logistic Regression
    # https://en.wikipedia.org/wiki/Linear_classifier
    # https://en.wikipedia.org/wiki/Logistic_regression
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000)


# ** **********************************************************
# Model selector (jump table style)
# ** **********************************************************


def get_model(name: str, tree_depth: int):
    if name == "SVM":
        return load_svm()
    elif name == "DecisionTree":
        return load_decision_tree(tree_depth)
    elif name == "TreeStump":
        return load_decision_tree(1)
    elif name == "LinearModel":
        return load_linear()
    else:
        raise ValueError(
            "Model must be one of: SVM | DecisionTree | TreeStump | LinearModel"
        )


# ** **********************************************************
# CLI
# ** **********************************************************


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate ML models on Vacuum World dataset"
    )

    parser.add_argument(
        "-f", "--file", required=True, help="CSV dataset (must include header row)"
    )

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        choices=["SVM", "DecisionTree", "TreeStump", "LinearModel"],
        help="Model type",
    )

    parser.add_argument(
        "--visualize-tree",
        action="store_true",
        help="Visualize decision tree (only works for tree models)",
    )

    parser.add_argument(
        "--show-feature-importances",
        action="store_true",
        help="Print feature importances (tree-based models only)",
    )

    parser.add_argument(
        "--tree-depth",
        type=int,
        default=1,
        help="Max depth of decision tree (default=1, acts like a stump)",
    )

    parser.add_argument(
        "--make-graphviz-tree-file",
        action="store_true",
        help="Create a PDF file of a decision tree (decision_tree.pdf)",
    )

    parser.add_argument("--test-size", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ** **********************************************************
# Main function - begin operations
# ** **********************************************************


def main():
    args = parse_args()

    # Load CSV WITH HEADER
    df = pd.read_csv(args.file)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    feature_names = df.columns[:-1]  # Reads first line for feature names

    # Encode labels/classes (the DirtDistribution names)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=args.test_size, random_state=args.seed, stratify=y_enc
    )

    # Get a Model and then train it
    clf = get_model(args.model, args.tree_depth)
    clf.fit(X_train, y_train)

    # Predict/classify against the testing data
    y_pred = clf.predict(X_test)

    # ** ******************************************************
    # Evaluation
    # ** ******************************************************

    print("\nModel Used:", args.model)
    acc_pct = 100 * accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc_pct:.1f}% || ", accuracy_score(y_test, y_pred))

    # Print out the confusion matrix
    print("\n" + "-" * 78)
    cm = confusion_matrix(y_test, y_pred)

    df_cm = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    print("\nConfusion Matrix:\n")
    print(df_cm)

    # Print out a classification report for all classes
    print("\n" + "-" * 78)
    print("\nClassification Report:\n")
    print(
        classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    )

    # ** ******************************************************
    # Feature importances
    # This shows which features had the most decision power in a decision tree
    # Basically, which ones had the most entropy reduction
    # ** ******************************************************

    if args.show_feature_importances:
        if hasattr(clf, "feature_importances_"):
            print("\nFeature Importances:\n")
            for name, val in zip(feature_names, clf.feature_importances_):
                print(f"{name}: {val:.4f}")
        else:
            print("\nThis model does not support feature_importances_")

    # ** ******************************************************
    # Tree visualization
    # Uses the built-in scikit tools to show the tree
    # Works well for 1..3 levels of tree
    # ** ******************************************************

    if args.visualize_tree:
        if args.model in ["DecisionTree", "TreeStump"]:
            try:
                from sklearn.tree import plot_tree
                import matplotlib.pyplot as plt

                plt.figure(figsize=(20, 10))

                plot_tree(
                    clf,
                    filled=True,
                    feature_names=feature_names,
                    class_names=le.classes_,
                    rounded=True,
                )

                plt.title(f"Decision Tree Visualization ({args.model})")
                plt.show()

            except Exception as e:
                print("Tree visualization failed:", e)
        else:
            print("--visualize-tree only works for DecisionTree or TreeStump")

    # ** ******************************************************
    # ** Output a PDF of the tree using graphviz (if desired)
    # ** ******************************************************
    if args.make_graphviz_tree_file:
        from sklearn.tree import export_graphviz
        import graphviz

        dot_data = export_graphviz(
            clf,
            out_file=None,
            feature_names=[f"f{i}" for i in range(X.shape[1])],
            class_names=le.classes_,
            filled=True,
            rounded=True,
            special_characters=True,
        )

        graph = graphviz.Source(dot_data)
        graph.render(filename="decision_tree", format="pdf", cleanup=True)


if __name__ == "__main__":
    main()

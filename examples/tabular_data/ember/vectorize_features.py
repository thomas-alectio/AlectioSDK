import os
import json
import ember
import argparse


def main():
    prog = "train_ember"
    descr = "Train an ember model from a directory with raw feature files"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("-v", "--featureversion", type=int, default=2, help="EMBER feature version")
    parser.add_argument("datadir", metavar="DATADIR", type=str, help="Directory with raw features")
    args = parser.parse_args()

    if not os.path.exists(args.datadir) or not os.path.isdir(args.datadir):
        parser.error("{} is not a directory with raw feature files".format(args.datadir))

    X_train_path = os.path.join(args.datadir, "X_train.dat")
    y_train_path = os.path.join(args.datadir, "y_train.dat")
    if not (os.path.exists(X_train_path) and os.path.exists(y_train_path)):
        print("Creating vectorized features")
        ember.create_vectorized_features(args.datadir, args.featureversion)


if __name__ == "__main__":
    main()

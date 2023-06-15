import argparse
from sklearn.datasets import make_blobs

SEED = 42


def generate_cluster_data(
    n_samples=10000, n_features=2, n_centers=20, out_path="data/data.txt", SEED=42
):
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_centers,
        cluster_std=0.1,
        random_state=SEED,
    )

    for file in [
        "data.txt",
        f"backup_data/data_{n_samples}_{n_features}_{n_centers}.txt",
    ]:
        with open(file, "w") as f:
            for idx, (feature, label) in enumerate(zip(X, y)):
                for data in feature:
                    f.write(f"{data} ")
                f.write(str(label))

                if idx != len(X) - 1:
                    f.write("\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_samples", type=int, default=10000)
    argparser.add_argument("--n_features", type=int, default=2)
    argparser.add_argument("--n_centers", type=int, default=20)
    args = argparser.parse_args()

    generate_cluster_data(
        n_samples=args.n_samples, n_features=args.n_features, n_centers=args.n_centers
    )

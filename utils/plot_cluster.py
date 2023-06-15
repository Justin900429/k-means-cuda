import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--data", type=str, default="data/answer.txt")
    args = argparse.parse_args()

    cluster_x = []
    cluster_y = []
    data_x = []
    data_y = []
    real_label = []
    predict = []

    with open(args.data, "r") as f:
        elapsedTime = float(f.readline().strip())
        clusters, num_data = [int(data) for data in f.readline().strip().split()[:2]]
        for _ in range(clusters):
            x, y = [float(data) for data in f.readline().strip().split()[:2]]
            cluster_x.append(x)
            cluster_y.append(y)
        for _ in range(num_data):
            data = f.readline().strip().split()
            data_x.append(float(data[0]))
            data_y.append(float(data[1]))
            real_label.append(int(data[2]))
            predict.append(int(data[3]))

    print("Score with ground-truth", adjusted_rand_score(real_label, predict))
    print("Number of unique clusters", len(np.unique(predict)))
    print(f"Execution Time: {elapsedTime}ms")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Real label")
    plt.scatter(data_x, data_y, c=real_label)
    plt.subplot(1, 2, 2)
    plt.title("Predict label")
    plt.scatter(data_x, data_y, c=predict)
    plt.scatter(cluster_x, cluster_y, c="black", marker="x", s=100)
    plt.show()

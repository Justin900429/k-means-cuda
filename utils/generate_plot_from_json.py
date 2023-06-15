import argparse
import matplotlib.pyplot as plt
import scienceplots
import os
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="benchmark.json")
    args = parser.parse_args()

    with open(args.data, "r") as f:
        data = json.load(f)

    os.makedirs("plot", exist_ok=True)
    label_list = ["$CPU$", "${GPU}^*$", "$GPU$"]

    for group_idx, (group_name, group) in enumerate(data.items()):
        title = f"Time under different {group_name}"
        param_label = []
        data_list = []
        for param in group:
            param_label.append(
                param["parameters"]["cluster"]
                if "cluster" in group_name
                else param["parameters"]["num_data"]
            )
            data_list.append(list(param["execution_time"].values()))

        with plt.style.context(["science", "nature"]):
            plt.figure()
            for idx, single_data in enumerate(zip(*data_list)):
                plt.plot(range(len(single_data)), single_data, label=label_list[idx])
            plt.title(title)
            plt.xticks(range(len(param_label)), param_label)
            plt.legend()
            plt.xlabel(
                "Number of clusters" if "cluster" in group_name else "Number of data"
            )
            plt.ylabel("Execution Time (ms)")
            plt.savefig(
                f"plot/{'_'.join(group_name.split(' '))}.png",
                dpi=1000,
                transparent=True,
            )

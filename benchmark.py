import sys
import time
import json
import subprocess

from colored import fg, attr

from utils.generate_cluster import generate_cluster_data


name_to_exec = {
    "ck": "cpu_kmeans",
    "gkori": "gpu_stride_kmeans_ori",
    "gkimp": "gpu_stride_kmeans_imp",
}


class Benchmark:
    def __init__(self, root="./", test_groups={"test": [(20, 10000)]}):
        self.root = root
        self._make_file()
        self._run_group(test_groups)

    def _run_subprocess(self, cmd=None):
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def _output_stream(self, output_str):
        sys.stdout.write(output_str)
        sys.stdout.flush()

    def _make_file(self):
        count = 1
        with self._run_subprocess(["make", "compare"]) as proc:
            while proc.poll() is None:
                self._output_stream(
                    f"{attr('bold')}\033[2K\033[1GCompile and making the executable files{'.' * count}"
                )
                count = (count + 1) % 4
                time.sleep(1)
        self._output_stream(f" {fg('blue')}Done!{attr(0)}\n")

    def _read_cluster_file(self, file_path):
        cluster_x = []
        cluster_y = []
        predict = []

        with open(file_path, "r") as f:
            time = float(f.readline().strip())
            clusters, num_data = [
                int(data) for data in f.readline().strip().split()[:2]
            ]
            for _ in range(clusters):
                x, y = [float(data) for data in f.readline().strip().split()[:2]]
                cluster_x.append(x)
                cluster_y.append(y)
            for _ in range(num_data):
                data = f.readline().strip().split()
                predict.append(int(data[3]))

        return time, cluster_x, cluster_y, predict

    def _check_all_the_same(self, predict_results):
        for idx in range(len(predict_results) - 1):
            if (len(predict_results[idx]) != len(predict_results[idx + 1])) or any(
                [
                    predict_results[idx][ele] != predict_results[idx + 1][ele]
                    for ele in range(len(predict_results[0]))
                ]
            ):
                return False
        return True

    def end(self):
        run_exectute = False
        count = 1
        with self._run_subprocess(["make", "clean"]) as proc:
            while proc.poll() is None:
                run_exectute = True
                self._output_stream(
                    f"\n{attr('bold')}\033[2K\033[1GCleaning up the files{'.' * count}{attr(0)}"
                )
                count = (count + 1) % 4
                time.sleep(0.7)
            output = proc.stdout.read().decode("utf-8")

        if run_exectute:
            self._output_stream("\033[F")
        self._output_stream(
            f"\n{attr('bold')}**Finish cleaning up the files! The following commands were executed:**{attr(0)}\n"
        )
        self._output_stream("> " + f"\033[3m{output}{attr(0)}\n")

    def _run_stage(self, num_cluster, num_data):
        generate_cluster_data(n_samples=num_data, n_centers=num_cluster, save_backup=True)
        predict_result = []
        elapsedTime = []
        for exec_type in sorted(list(name_to_exec.keys())):
            count = 1
            with self._run_subprocess(
                [f"{self.root}{name_to_exec[exec_type]}", str(num_cluster)]
            ) as proc:
                while proc.poll() is None:
                    self._output_stream(
                        f"\033[2K\033[1G({attr('underlined')}cluster: {num_cluster}{attr(0)}, {attr('underlined')}samples: {num_data}{attr(0)}) Running the command > {self.root}{name_to_exec[exec_type]} {num_cluster}{'.' * count}"
                    )
                    count = (count + 1) % 4
                    time.sleep(0.7)
                elapsed, *_, predict = self._read_cluster_file(
                    f"output/{name_to_exec[exec_type]}_{num_cluster}_{num_data}.txt"
                )
                elapsedTime.append(elapsed)
                predict_result.append(predict)

        self._output_stream(
            f"\033[2K\033[1GFinish running the commands for {attr('underlined')}cluster: {num_cluster}{attr(0)}, {attr('underlined')}samples: {num_data}{attr(0)}. Result:"
        )

        if self._check_all_the_same(predict_result):
            passed = True
            self._output_stream(f" {fg('green')}Passed!{attr(0)}\n")
        else:
            passed = False
            self._output_stream(f" {fg('red')}Failed!{attr(0)}\n")

        return elapsedTime, passed

    def _run_stages(self, test_stages):
        test_results = []
        count_error = 0
        for test_stage in test_stages:
            elapsed_time, passed = self._run_stage(*test_stage)
            test_results.append(
                {
                    "parameters": {
                        "cluster": test_stage[0],
                        "num_data": test_stage[1],
                    },
                    "execution_time": {
                        key: elapsed_time[idx]
                        for idx, key in enumerate(sorted(list(name_to_exec.keys())))
                    },
                    "passed": passed,
                }
            )
            count_error += not passed
        return test_results, count_error

    def _run_group(self, test_groups):
        dump = {}
        for test_stages_name, test_stages in test_groups.items():
            left_space = (100 - len(test_stages_name)) // 2
            right_space = 100 - len(test_stages_name) - left_space
            self._output_stream(
                f"\n{fg('orange_1')}{left_space * '='}{attr('bold')}{test_stages_name}{attr(0)}{fg('orange_1')}{right_space * '='}{attr(0)}\n"
            )
            test_results, count_errors = self._run_stages(test_stages)
            dump[test_stages_name] = test_results

            if count_errors:
                output = (
                    f"{len(test_stages) - count_errors} passed, {count_errors} failed!"
                )
            else:
                output = f"{len(test_stages)} passed!"

            left_space = (100 - len(output)) // 2
            right_space = 100 - len(output) - left_space
            if count_errors:
                self._output_stream(
                    f"{fg('red')}{left_space * '='}{attr(0)}{attr('bold')}{fg('light_red')}{output}{attr(0)}{fg('red')}{right_space * '='}{attr(0)}\n"
                )
            else:
                self._output_stream(
                    f"{fg('green')}{left_space * '='}{attr(0)}{attr('bold')}{fg('light_green')}{output}{attr(0)}{fg('green')}{right_space * '='}{attr(0)}\n"
                )

        with open("benchmark.json", "w") as f:
            json.dump(dump, f, indent=2)

        self._output_stream(
            f"\n{attr('bold')}All tests are done! The results are saved in {fg('blue')}benchmark.json{attr(0)}{attr('bold')}.{attr(0)}\n"
        )


if __name__ == "__main__":
    test_groups = {
        "clusters": [
            (10, 10000),
            (50, 10000),
            (100, 10000),
            (500, 10000),
            (1000, 10000),
            (5000, 10000),
            (10000, 10000)
        ],
        "number of data": [
            (100, 100),
            (100, 500),
            (100, 1000),
            (100, 5000),
            (100, 10000),
            (100, 50000),
            (100, 100000)
        ],
    }
    bench_mark = Benchmark(test_groups=test_groups)
    bench_mark.end()

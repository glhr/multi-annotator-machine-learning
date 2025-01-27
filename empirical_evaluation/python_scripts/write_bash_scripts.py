import numpy as np
import os
from itertools import product


def generate_random_configs(hyperparam_options, n_configs=5, seed=0):
    """
    Generate random hyperparameter configurations from user-provided lists.

    Parameters
    ----------
    hyperparam_options : dict
        A dictionary with hyperparameter names as keys and lists of possible
        values to sample from as values. For example:

        {
            "batch_size": [16, 32, 64],
            "optimizer": ["sgd", "adam"],
            "learning_rate": [0.0001, 0.001, 0.01, 0.1]
        }

    n_configs : int, optional
        Number of random configurations to generate. Default is 5.

    Returns
    -------
    list of dict
        A list of hyperparameter configurations, each represented as a
        dictionary with hyperparameter names as keys and sampled values
        as values.
    """
    configs = []
    random_state = np.random.RandomState(seed)
    for _ in range(n_configs):
        config = {}
        for hyperparam_name, choices in hyperparam_options.items():
            config[hyperparam_name] = random_state.choice(choices)
        configs.append(config)
    return configs


def write_commands(
    config_combs: list,
    path_python_file: str = ".",
    directory: str = ".",
    use_slurm: bool = True,
    mem: str = "20gb",
    max_n_parallel_jobs: int = 12,
    cpus_per_task: int = 4,
    slurm_logs_path: str = "slurm_logs",
):
    """
    Writes Bash scripts for the experiments.

    Parameters
    ----------
    config_combs : list
        A list of dictionaries defining the configurations of the experiments.
    path_python_file : str, default="."
        Absolute path to the Python file to be executed.
    directory : str
        Path to the directory where the Bash scripts are to be saved.
    use_slurm : bool
        Flag whether SLURM shall be used.
    mem : str
        RAM size allocated for each experiment. Only used if `use_slurm=True`.
    max_n_parallel_jobs : int
        Maximum number of experiments executed in parallel. Only used if `use_slurm=True`.
    cpus_per_task : int
        Number of CPUs allocated for each experiment. Only used if `use_slurm=True`.
    use_gpu : bool
        Flag whether to use a GPU. Only used if `use_slurm=True`.
    slurm_logs_path : str
        Path to the directory where the SLURM logs are to saved. Only used if `use_slurm=True`.
    """
    for cfg_dict in config_combs:
        permutations_dicts = cfg_dict.pop("params")
        if isinstance(permutations_dicts, dict):
            keys, values = zip(*permutations_dicts.items())
            permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
        n_jobs = len(permutations_dicts)
        if max_n_parallel_jobs > n_jobs:
            max_n_parallel_jobs = n_jobs
        job_name = cfg_dict['experiment_name']
        filename = os.path.join(directory, f"{job_name}.sh")
        commands = [
            f"#!/usr/bin/env bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --array=1-{n_jobs}%{max_n_parallel_jobs}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --ntasks=1",
            f"#SBATCH --get-user-env",
            f"#SBATCH --time=12:00:00",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --partition=main",
            f"#SBATCH --output={slurm_logs_path}/{job_name}_%A_%a.log",
        ]
        if cfg_dict["accelerator"] == "gpu":
            commands += [
                f"#SBATCH --gres=gpu:1",
                f'eval "$(sed -n "$(($SLURM_ARRAY_TASK_ID+{13})) p" {filename})"',
                f"exit 0",
            ]
        else:
            commands += [
                f'eval "$(sed -n "$(($SLURM_ARRAY_TASK_ID+{12})) p" {filename})"',
                f"exit 0",
            ]
        python_command = f"srun python"
        if not use_slurm:
            commands = [commands[0]]
            python_command = f"python"
        for param_dict in permutations_dicts:
            commands.append(f"{python_command} {path_python_file} ")
            for k, v in (cfg_dict | param_dict).items():
                commands[-1] += f"{k}={v} "
            if not use_slurm:
                commands.append("wait")
        print(f"{filename}: {n_jobs}")
        with open(filename, "w") as f:
            for item in commands:
                f.write("%s\n" % item)


if __name__ == "__main__":
    # TODO: Update the default arguments of the `write_commands` function below to fit your machine.
    path_python_file = "your/absolute/path/to/perform_experiment.py"
    directory = "."
    data_sets_path = "your/absolute/path/to/data_sets"
    use_slurm = True
    mem = "10gb"
    max_n_parallel_jobs = 100
    cpus_per_task = 4
    accelerator = "cpu"
    slurm_logs_path = ""

    seed_list = list(range(5))
    classifiers = [
        "annot_mix",
        #"conal",
        #"crowd_layer",
        #"crowdar",
        #"ground_truth",
        #"madl",
        #"majority_vote",
        #"trace_reg",
        #"geo_reg_f",
        #"geo_reg_w",
        #"union_net",
    ]

    # Define search spaces for hyperparameters.
    hp_ranges = {
        "data.optimizer.gt_params.lr": np.logspace(-4, -1, 10),
        "data.optimizer.gt_params.weight_decay": np.logspace(-6, -3, 10),
        "data.optimizer.ap_params.lr": np.logspace(-4, -1, 10),
        "data.optimizer.ap_params.weight_decay": np.logspace(-6, -3, 10),
        "data.train_batch_size": np.array([16, 32, 64, 128])
    }
    global_seed_offset = 0

    def get_params(seed_offset=0):
        params = []
        for seed in seed_list:
            for clf in classifiers:
                hpc_addendum = {
                    "seed": [seed],
                    "classifier": [clf],
                    "data.class_definition.realistic_split": [f"cv-5-{seed}"]
                }
                hpc = hp_ranges | hpc_addendum
                params += generate_random_configs(hyperparam_options=hpc, n_configs=20, seed=seed_offset+seed)
        return params

    # Datasets: variants of dopanim
    for variant in ["worst-1", "worst-2", "worst-var", "rand-1", "rand-2", "rand-var", "full"]:
        config_combs = [
            {
                "experiment_name": f"hyperparameter_search_dopanim_{variant}",
                "data": "dopanim",
                "data.class_definition.variant": variant,
                "data.class_definition.root": data_sets_path,
                "architecture": "dino_head",
                "ssl_model": "dino_backbone",
                "accelerator": accelerator,
                "params": get_params(global_seed_offset*100)
            },
        ]
        write_commands(
            path_python_file=path_python_file,
            directory=directory,
            config_combs=config_combs,
            slurm_logs_path=slurm_logs_path,
            max_n_parallel_jobs=max_n_parallel_jobs,
            mem=mem,
            cpus_per_task=cpus_per_task,
            use_slurm=use_slurm,
        )
        global_seed_offset += 1

    # Dataset: label-me
    config_combs = [
        {
            "experiment_name": f"hyperparameter_search_label_me",
            "data": "label_me",
            "data.class_definition.root": data_sets_path,
            "architecture": "dino_head",
            "ssl_model": "dino_backbone",
            "accelerator": accelerator,
            "params": get_params(global_seed_offset*100)
        },
    ]
    write_commands(
        path_python_file=path_python_file,
        directory=directory,
        config_combs=config_combs,
        slurm_logs_path=slurm_logs_path,
        max_n_parallel_jobs=max_n_parallel_jobs,
        mem=mem,
        cpus_per_task=cpus_per_task,
        use_slurm=use_slurm,
    )

    # Datasets: music_genres and sentiment_polarity
    for ds in ["music_genres", "sentiment_polarity"]:
        config_combs = [
            {
                "experiment_name": f"hyperparameter_search_{ds}",
                "data": ds,
                "data.class_definition.root": data_sets_path,
                "architecture": f"tabnet_{ds}",
                "ssl_model": "none",
                "accelerator": accelerator,
                "params": get_params(global_seed_offset*100)
            },
        ]
        write_commands(
            path_python_file=path_python_file,
            directory=directory,
            config_combs=config_combs,
            slurm_logs_path=slurm_logs_path,
            max_n_parallel_jobs=max_n_parallel_jobs,
            mem=mem,
            cpus_per_task=cpus_per_task,
            use_slurm=use_slurm,
        )
        global_seed_offset += 1

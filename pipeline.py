import multiprocessing
import queue
import yaml
import os
import subprocess
import json
import time
import pynvml

def get_free_gpu(min_memory_required):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    free_memory = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if mem_info.free > min_memory_required:
            free_memory.append((i, mem_info.free))

    # Sort GPUs by free memory in descending order
    free_memory.sort(key=lambda x: x[1], reverse=True)
    
    # Return the GPU index with sufficient memory
    return free_memory[0][0] if free_memory else None

def set_gpu_device(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def generate_hyperparameter_sets():
    # Example hyperparameter sets
    hyperparameter_sets = [
        {'lr': 1e-4, 'batch_size': 32, 'weight_decay': 1e-5, 'num_epochs': 10,  'seed':0, 'output_dir':10}
        # {'lr': 1e-3, 'batch_size': 32, 'weight_decay': 1e-4, 'num_epochs': 20,  'seed':1, 'output_dir':11}
        # {'lr': 1e-2, 'batch_size': 32, 'weight_decay': 1e-3, 'num_epochs': 30, 'seed':2, 'output_dir':12},
        # Add more hyperparameter sets as needed
    ]
    return hyperparameter_sets

def update_params_yaml(params, experiment):
    params['train']['experiment'] = experiment

    with open(f'params_{experiment}.yaml', 'w') as file:
        yaml.dump(params, file)

def run_pipeline(experiment, hyperparams, min_memory_required):

    # Step 2: Train model and capture the output path
    gpu_id = None
    while gpu_id is None:
        gpu_id = get_free_gpu(min_memory_required)
        if gpu_id is None:
            print(f"No available GPU with {min_memory_required} bytes free memory. Waiting...")
            time.sleep(10)  # Wait for 10 seconds before checking again
        else:
            set_gpu_device(gpu_id)
            print(f"Using GPU {gpu_id} for training experiment {experiment}")

    train_command = (
        f'python3 src/train.py {experiment} {hyperparams["lr"]} {hyperparams["batch_size"]} '
        f'{hyperparams["weight_decay"]} {hyperparams["num_epochs"]} {hyperparams["seed"]} {hyperparams["output_dir"]}'
    )
    result = subprocess.run(train_command, shell=True, check=True, stdout=subprocess.PIPE)
    model_path = result.stdout.decode().strip()
    
    # Step 3: Test model using the output path from the training step
    gpu_id = None
    while gpu_id is None:
        gpu_id = get_free_gpu(min_memory_required)
        if gpu_id is None:
            print(f"No available GPU with {min_memory_required} bytes free memory. Waiting...")
            time.sleep(10)  # Wait for 10 seconds before checking again
        else:
            set_gpu_device(gpu_id)
            print(f"Using GPU {gpu_id} for testing experiment {experiment}")

    test_command = f'python3 src/test.py {experiment} {model_path} {hyperparams["lr"]} {hyperparams["batch_size"]} {hyperparams["weight_decay"]} {hyperparams["num_epochs"]} {hyperparams["seed"]} {hyperparams["output_dir"]}'
    result = subprocess.run(test_command, shell=True, check=True, stdout=subprocess.PIPE)
    results_path = result.stdout.decode().strip()

    return model_path, results_path

def aggregate_results(num_subjects):
    all_results = []
    for i in range(num_subjects):
        results_path = os.path.join('results', f'results_{i}.json')
        with open(results_path, 'r') as f:
            results = json.load(f)
            all_results.append(results)
    
    # Calculate the average performance metric
    average_performance = sum(result['performance_metric'] for result in all_results) / num_subjects
    print(f'Average Performance: {average_performance}')

def worker(experiment_queue, min_memory_required):
    while True:
        try:
            experiment, hyperparams = experiment_queue.get_nowait()
        except queue.Empty:
            break

        # # Load the base params
        # with open('params.yaml', 'r') as file:
        #     params = yaml.safe_load(file)

        # # Update params for this experiment
        # update_params_yaml(params, experiment)

        # Run the pipeline for the given experiment and hyperparams
        run_pipeline(experiment, hyperparams, min_memory_required)
        experiment_queue.task_done()

if __name__ == '__main__':
    num_subjects = 8  # Adjust according to your number of subjects
    min_memory_required = 4 * 1024 * 1024 * 1024  # 4 GB, adjust according to your needs

    hyperparameter_sets = generate_hyperparameter_sets()
    
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    
    experiments = list(params['prepare']['PAMAP'].keys())

    # Create a queue of experiments with different hyperparameters
    experiment_queue = multiprocessing.JoinableQueue()
    for experiment in experiments:
        for hyperparams in hyperparameter_sets:
            experiment_queue.put((experiment, hyperparams))

    # Start worker processes
    num_workers = min(multiprocessing.cpu_count(), len(hyperparameter_sets) * len(experiments))
    processes = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker, args=(experiment_queue, min_memory_required))
        processes.append(p)
        p.start()

    # Wait for all experiments to be processed
    experiment_queue.join()

    # Ensure all processes have finished
    for p in processes:
        p.join()

    # Aggregate the results
    # aggregate_results(num_subjects)

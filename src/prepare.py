from pamap import PAMAP
import sys
import yaml

import pandas as pd
import os
import yaml
import sys
import numpy as np

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def params(dataset_name, experiment_name):
    # Load parameters from params.yaml
    params = load_params()
    prepare_params = params['prepare']
    seed = prepare_params['seed']
    print(f"Seed: {seed}")
    experiment_params = prepare_params[dataset_name][experiment_name]
    train_subjects = experiment_params['train']
    validation_subjects = experiment_params['validation']
    test_subjects = experiment_params['test']

    
    print(f"Train subjects: {train_subjects}")
    print(f"Validation subjects: {validation_subjects}")
    print(f"Test subjects: {test_subjects}")

    return seed, train_subjects, validation_subjects,test_subjects

def save_data(output_dir, train_data, validation_data, test_data):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    train_arrays, train_labels = zip(*train_data)
    validation_arrays, validation_labels = zip(*validation_data)
    test_arrays, test_labels = zip(*test_data)

    train_path = os.path.join(output_dir, 'train_data.csv')
    validation_path = os.path.join(output_dir, 'validation_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')

    np.savez(os.path.join(output_dir, 'train_data.npz'), data=train_arrays, labels=train_labels)
    np.savez(os.path.join(output_dir, 'validation_data.npz'), data=validation_arrays, labels=validation_labels)
    np.savez(os.path.join(output_dir, 'test_data.npz'), data=test_arrays, labels=test_labels)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python prepare.py <dataset_name> <experiment_name>")
        sys.exit(1)
    dataset_name = sys.argv[1]
    experiment_name = sys.argv[2]
    output_dir = sys.argv[3]
    seed, train,validation,test = params(dataset_name, experiment_name)
    if dataset_name == "PAMAP":
        dataset = PAMAP(train=train, validation=validation, test= test )
        dataset.get_datasets()
        dataset.preprocessing()
        dataset.normalize()
        dataset.data_segmentation()
        dataset.prepare_dataset()
    new_train = [(a[0],a[1]) for a in dataset.training_final]
    new_val = [(a[0],a[1]) for a in dataset.validation_final]
    new_test = [(a[0],a[1]) for a in dataset.testing_final]
    print(len(new_train))
    print(len(new_val))
    print(len(new_test))

    # Save the processed data
    save_data(output_dir, new_train,new_val,new_test)
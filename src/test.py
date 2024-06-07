import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import f1_score
import os
import pandas as pd
import sys
import yaml

class Tester:
    def __init__(self, encoder, classifier, device):
        self.encoder = encoder
        self.classifier = classifier
        self.device = device

    def test(self,dataloader):
        self.encoder.eval()  # set the model to evaluation mode
        self.classifier.eval()
        correct_predictions = 0
        total_samples = 0
        targets_list = []
        predictions_list = []
        pid = "not_pid"

        with torch.no_grad():
            for data in dataloader:
                inputs, targets = data[0].to(self.device).float(), data[1].to(self.device)
                #print(inputs.shape)
                encoder = self.encoder(inputs)
                outputs_1 = self.classifier(encoder)
                predicted_classes = outputs_1.argmax(dim=1).squeeze()  # Find the class index with the maximum value in predicted
                correct_predictions += (predicted_classes == targets).sum().float()
                total_samples += targets.size(0)
                targets_list.append(targets)
                predictions_list.append(predicted_classes)
        pid = str(os.getpid())            
        cm = confusion_matrix(torch.cat(targets_list, dim=0).cpu().numpy(), torch.cat(predictions_list, dim = 0).cpu().numpy())
        data_to_save = np.concatenate((torch.cat(targets_list, dim=0).cpu().numpy()[:,np.newaxis],torch.cat(predictions_list, dim = 0).cpu().numpy()[:,np.newaxis]), axis=1)
        print("Total samples testing: ",total_samples)
        accuracy = correct_predictions / total_samples
        F1_macro = f1_score(torch.cat(targets_list, dim=0).cpu().numpy(), torch.cat(predictions_list, dim = 0).cpu().numpy(), average='macro')
        F1_w = f1_score(torch.cat(targets_list, dim=0).cpu().numpy(), torch.cat(predictions_list, dim = 0).cpu().numpy(), average='weighted')
        cr = classification_report(torch.cat(targets_list, dim=0).cpu().numpy(), torch.cat(predictions_list, dim = 0).cpu().numpy(), labels=[0,1,2,3,4,5,6,7,8,9,10,11], output_dict=True)
        print(f"Test F1: {F1_macro:.4f}")
        print(cr)
        return cm, data_to_save, cr

def load_data(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        arrays = data['data']
        labels = data['labels']
    return list(zip(arrays, labels))

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def main(data_dir, model_dir, output_directory, params):

    test_data = load_data(os.path.join(data_dir, 'test_data.npz'))
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers = 1)
    encoder = torch.load(model_dir+'/encoder.pt')
    classifier = torch.load(model_dir +'/classifier.pt')
    testerA = Tester(encoder, classifier, device= "cuda:0")
    cm, data, cr = testerA.test(test_dataloader)
    os.makedirs(os.path.dirname(output_directory +'classification_metrics/'), exist_ok=True)
    dataframe = pd.DataFrame.from_dict(cr)
    dataframe.to_csv(output_directory +'classification_metrics/results.csv')
    os.makedirs(os.path.dirname(output_directory +'confusion_matrix/'), exist_ok=True)
    np.savetxt(output_directory +'confusion_matrix/results.txt', cm)
    os.makedirs(os.path.dirname(output_directory +'prediction_target/'), exist_ok=True)
    np.savetxt(output_directory +'prediction_target/results.txt', data)
    with open(output_directory +'classification_metrics/'+'params.yaml', 'w') as file:
        yaml.dump(params, file)
if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("Usage: python train.py <data_dir> <model_output_path> <experiment_name>")
    #     sys.exit(1)
    experiment = sys.argv[1]
    data_dir = f"data/prepared/PAMAP/{experiment}/"
    model_dir = sys.argv[2]

    params = {"experiment":str(sys.argv[1]),
            "lr":float(sys.argv[3]),
            "batch_size":int(sys.argv[4]),
            "weight_decay":float(sys.argv[5]),
            "epochs":int(sys.argv[6]),
            "seed": int(sys.argv[7]),
            "output_dir":str(sys.argv[8])}

    output_directory = f"metrics/PAMAP/{experiment}/{params['output_dir']}/"
    main(data_dir, model_dir, output_directory, params)
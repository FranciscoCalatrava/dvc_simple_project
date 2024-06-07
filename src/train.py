
import torch
import torch.nn as nn
import numpy as np
import sys
import yaml
import pandas as pd
import os
from model import Feature_Extractor, Classifier
import joblib
from torch.utils.data import DataLoader
import torch.optim as optim


class Trainer():
    def __init__(self, encoder, decoder, train_dataloader, val_dataloader, device, optimizer, criterion, embeddings = False) -> None:
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        self.val_dataloader = val_dataloader

        self.embeddings_model = embeddings

        # self.embeddings = torch.tensor(np.load('weights_non_normalized.npy')).float().to(device="cuda:0")
        self.embeddings = torch.randn(3,300).to(device)

        self.train_dataloader = train_dataloader
        self.device = device
        self.optimizer_reconstruction = optimizer["classifier"]
        self.criterion_reconstruction = criterion["classifier"]
    
    def model(self, encoder, decoder):
        return lambda x: decoder(encoder(x))
    
    def frozen(self,model):
        for a in model.parameters():
            a.requires_grad = False

    def loss_function(self, output_classifier, target):
        vector_embedding = torch.matmul(output_classifier,self.embeddings)

        # print(vector_embedding.shape)
        target_embedding = torch.zeros((target.size(0),self.embeddings.size(1))).to(self.device)
        for a in range(target.size(0)):
            target_embedding[a,:] =self.embeddings[target[a],:]
        return nn.MSELoss()(vector_embedding,target_embedding)



    
    def train_one_epoch_emebddings(self):
        running_loss_reconstruction = 0.0

        samples_per_epoch = 0
        correct_predictions = {'classifier': 0}
        prediction_list = {'classifier': []}
        target_list = {'classifier':[]}

        accuracy = {'classifier': 0.0}


        avg_loss_reconstruction = 0.0

        self.encoder.train(True)
        self.decoder.train(False)

        softmax = nn.Softmax()
        
        for i, data in enumerate(self.train_dataloader):
            inputs, activity_class = data

            inputs = inputs.to(self.device).float()
            activity_class = activity_class.to(self.device)



            outputs_encoder = self.encoder(inputs)
            outputs_reconstruction = self.decoder(outputs_encoder)

            # loss_reconstruction = self.criterion_reconstruction(outputs_reconstruction, activity_class)
            loss_reconstruction = self.loss_function(outputs_reconstruction,activity_class)


            self.decoder.zero_grad()
            self.encoder.zero_grad()
            loss_reconstruction.backward()
            self.optimizer_reconstruction.step()

            running_loss_reconstruction += loss_reconstruction.item()
            predictions_classifier = torch.argmax(softmax(outputs_reconstruction), dim = 1)
            correct_predictions["classifier"] += (torch.squeeze(predictions_classifier) == activity_class).sum()
            prediction_list["classifier"].append(torch.squeeze(predictions_classifier))
            target_list["classifier"].append(activity_class)
            samples_per_epoch += activity_class.size(0)
        avg_loss_reconstruction = running_loss_reconstruction/samples_per_epoch
        accuracy['classifier'] = correct_predictions["classifier"].item()/samples_per_epoch
        return avg_loss_reconstruction, accuracy['classifier']
    
    def validation_embeddings(self):
        self.encoder.eval()
        self.decoder.eval()
        running_validation_loss_reconstruction = 0.0
        samples = 0
        avg_validation_loss_reconstruction = 0
        softmax = nn.Softmax()
        correct_predictions = {'classifier': 0}
        prediction_list = {'classifier': []}
        target_list = {'classifier':[]}
        accuracy = {'classifier': 0.0}
        with torch.no_grad():
            for i, vdata in enumerate(self.val_dataloader):
                inputs, activity_class = vdata
                inputs = inputs.to(self.device).float()
                activity_class = activity_class.to(self.device)

                outputs_encoder = self.encoder(inputs)
                outputs_reconstruction = self.decoder(outputs_encoder)

                validation_loss_reconstruction = self.criterion_reconstruction(outputs_reconstruction,activity_class)
                running_validation_loss_reconstruction += validation_loss_reconstruction.item()
                predictions_classifier = torch.argmax(softmax(outputs_reconstruction), dim = 1)
                correct_predictions["classifier"] += (torch.squeeze(predictions_classifier) == activity_class).sum()
                prediction_list["classifier"].append(torch.squeeze(predictions_classifier))
                target_list["classifier"].append(activity_class)
                samples += activity_class.size(0)
            avg_validation_loss_reconstruction = running_validation_loss_reconstruction/samples
            accuracy['classifier'] = correct_predictions["classifier"].item()/samples
        return avg_validation_loss_reconstruction,accuracy['classifier']
    

    def train_one_epoch(self):
        running_loss_reconstruction = 0.0

        samples_per_epoch = 0
        correct_predictions = {'classifier': 0}
        prediction_list = {'classifier': []}
        target_list = {'classifier':[]}

        accuracy = {'classifier': 0.0}


        avg_loss_reconstruction = 0.0

        self.encoder.train(True)
        self.decoder.train(False)

        softmax = nn.Softmax()
        
        for i, data in enumerate(self.train_dataloader):
            inputs, activity_class = data

            inputs = inputs.to(self.device).float()
            activity_class = activity_class.to(self.device)



            outputs_encoder = self.encoder(inputs)
            outputs_reconstruction = self.decoder(outputs_encoder)

            loss_reconstruction = self.criterion_reconstruction(outputs_reconstruction, activity_class)
            # loss_reconstruction = self.loss_function(outputs_reconstruction,activity_class)


            self.decoder.zero_grad()
            self.encoder.zero_grad()
            loss_reconstruction.backward()
            self.optimizer_reconstruction.step()

            running_loss_reconstruction += loss_reconstruction.item()
            predictions_classifier = torch.argmax(softmax(outputs_reconstruction), dim = 1)
            correct_predictions["classifier"] += (torch.squeeze(predictions_classifier) == activity_class).sum()
            prediction_list["classifier"].append(torch.squeeze(predictions_classifier))
            target_list["classifier"].append(activity_class)
            samples_per_epoch += activity_class.size(0)
        avg_loss_reconstruction = running_loss_reconstruction/samples_per_epoch
        accuracy['classifier'] = correct_predictions["classifier"].item()/samples_per_epoch
        return avg_loss_reconstruction, accuracy['classifier']
    

    def validation(self):
        self.encoder.eval()
        self.decoder.eval()
        running_validation_loss_reconstruction = 0.0
        samples = 0
        avg_validation_loss_reconstruction = 0
        softmax = nn.Softmax()
        correct_predictions = {'classifier': 0}
        prediction_list = {'classifier': []}
        target_list = {'classifier':[]}
        accuracy = {'classifier': 0.0}
        with torch.no_grad():
            for i, vdata in enumerate(self.val_dataloader):
                inputs, activity_class = vdata
                inputs = inputs.to(self.device).float()
                activity_class = activity_class.to(self.device)

                outputs_encoder = self.encoder(inputs)
                outputs_reconstruction = self.decoder(outputs_encoder)

                validation_loss_reconstruction = self.criterion_reconstruction(outputs_reconstruction,activity_class)
                running_validation_loss_reconstruction += validation_loss_reconstruction.item()
                predictions_classifier = torch.argmax(softmax(outputs_reconstruction), dim = 1)
                correct_predictions["classifier"] += (torch.squeeze(predictions_classifier) == activity_class).sum()
                prediction_list["classifier"].append(torch.squeeze(predictions_classifier))
                target_list["classifier"].append(activity_class)
                samples += activity_class.size(0)
            avg_validation_loss_reconstruction = running_validation_loss_reconstruction/samples
            accuracy['classifier'] = correct_predictions["classifier"].item()/samples
        return avg_validation_loss_reconstruction,accuracy['classifier']
    
    def train(self, epochs):

        for a in range(epochs):
            if self.embeddings_model == False:
                train_loss, train_accuracy = self.train_one_epoch()
                val_loss, val_accuracy = self.validation()
            elif self.embeddings_model == True:
                train_loss, train_accuracy = self.train_one_epoch_emebddings()
                val_loss, val_accuracy = self.validation_embeddings()
            # print(f"train accuracy: {train_accuracy} || val accuracy {val_accuracy}")
            # print(f"train loss: {train_loss} || val loss {val_loss}")
    
        return self.encoder, self.decoder
    
def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_data(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        arrays = data['data']
        labels = data['labels']
    return list(zip(arrays, labels))

def main(data_dir, model_output_path, params):
    seed = params['seed']
    lr = params['lr']
    bs = params['batch_size']
    weight_decay = params['weight_decay']
    participant = params['experiment']
    id_experiment = params['output_dir']


    train_data = load_data(os.path.join(data_dir+participant+'/', 'train_data.npz'))
    validation_data = load_data(os.path.join(data_dir+participant+'/', 'validation_data.npz'))


    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True, num_workers = 1)
    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=bs, shuffle=True, num_workers = 1)


    model = Feature_Extractor(input_shape = (64,1,128) , num_blocks = [2,2,0,0], in_channel = 18, seed = seed).to("cuda:0")
    classifier = Classifier(12,seed)
    optimizer = {'classifier': optim.Adam(list(model.parameters())+list(classifier.parameters()) , lr= float(lr), weight_decay= float(weight_decay))}
    criterion = {"classifier": nn.CrossEntropyLoss()}

    trainerB = Trainer(model,classifier, train_dataloader, validation_dataloader, "cuda:0", optimizer, criterion)
    encoder, decoder = trainerB.train(2)


    os.makedirs(os.path.dirname(model_output_path+params['experiment']+ f"/{id_experiment}/"), exist_ok=True)
    # Save the trained model
    torch.save(encoder,model_output_path+params['experiment']+ f"/{id_experiment}/"+'encoder.pt')
    torch.save(decoder,model_output_path+params['experiment']+f"/{id_experiment}/"+'classifier.pt')
    with open(model_output_path+params['experiment']+f"/{id_experiment}/"+'params.yaml', 'w') as file:
        yaml.dump(params, file)
    print(model_output_path+params['experiment']+f"/{id_experiment}/")






if __name__ == "__main__":
    params = {"experiment":str(sys.argv[1]),
            "lr":float(sys.argv[2]),
            "batch_size":int(sys.argv[3]),
            "weight_decay":float(sys.argv[4]),
            "epochs":int(sys.argv[5]),
            "seed": int(sys.argv[6]),
            "output_dir":str(sys.argv[7])}

    
    data_dir = "data/prepared/PAMAP/"
    model_output_path = "model/PAMAP/"

    main(data_dir, model_output_path, params)
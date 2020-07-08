import pandas as pd
import os
import numpy as np
import torch
from torch.utils import data
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import f1_score as f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
import os

from sklearn.metrics import precision_recall_curve, auc, log_loss

def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

def compute_prauc(pred, gt):
    prec, recall, thresh = precision_recall_curve(gt, pred)
    prauc = auc(recall, prec)
    return prauc
def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr
def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0



import torch
from torch.utils import data

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels,values):
        'Initialization'
        self.labels = labels
        self.values = values
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.tensor(self.values[ID])
        #X = torch.tensor(np.array(X, dtype=np.float32))
        y = self.labels[ID]
        return X, y


def load_c3d(feature_file):
    with open(feature_file, 'r') as f:
        features = {}
        for line in f:
            key, value = line.rstrip('\n').split('\t')
            features[key] = np.array([x for x in value.split(' ')], np.float)
    return features

video_folder='me17in-devset-2'#'me17in-devset-3'#'me17in-devset-4'
annotations_train=pd.read_csv('me17in-devset-1/devset/annotations/devset-2017-video.txt',names=['video','interval','label','something1','something2'])
annotations_test=pd.read_csv('testset-2017-video.txt',names=['video','interval','label','something1','something2'])
#print(annotations)

video_names=[os.path.join(video_folder+'/devset/videos/',x) for x in os.listdir(video_folder+'/devset/videos')]
c3d_path_train=[os.path.join('me17in-devset-5/devset/features/Features_From_ETHZurich/c3d/',x) for x in os.listdir('me17in-devset-5/devset/features/Features_From_ETHZurich/c3d/')]
c3d_path_test=[os.path.join('me17in-testset-2/testset/features/Features_From_ETHZurich/c3d/',x) for x in os.listdir('me17in-testset-2/testset/features/Features_From_ETHZurich/c3d/')]
c3d_files_train=[x.split('c3d.txt')[0] for x in os.listdir('me17in-devset-5/devset/features/Features_From_ETHZurich/c3d/')]
c3d_files_test=[x.split('c3d.txt')[0] for x in os.listdir('me17in-testset-2/testset/features/Features_From_ETHZurich/c3d/')]

def loading_data(c3d_path, c3d_files,annotations):
    df_c3=pd.DataFrame()
    for i in range(len(c3d_path)):
        features=load_c3d(c3d_path[i])
        df=pd.DataFrame([(b, c) for a in [features] for b, c in a.items()],
                       columns=['keys','values'])
        #print(features_test)
        #print(df)
        df['keys+video']=c3d_files[i]+df['keys']
        ####NOTE TO ALISON FIX THIS SHIT
        df_c3=df_c3.append(df,ignore_index=True)
    annotations['keys+video'] = annotations['video'] + '-' + annotations['interval']
    df=pd.merge(df_c3,annotations, on='keys+video')
    print(df)
    IDs  = df.index
    print(df['label'].value_counts())
    labels=df['label'].values
    values=df['values'].apply(lambda s: list(map(float, s)))
    return IDs,labels,values


train_IDs,train_labels, train_values=loading_data(c3d_path_train,c3d_files_train,annotations_train)
test_IDs,test_labels, test_values=loading_data(c3d_path_test,c3d_files_test,annotations_test)

print(len(train_IDs))
print(len(test_IDs))

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        torch.manual_seed(41)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        # extra layers layers
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)

        return logits
class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        torch.manual_seed(41)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lstm = nn.LSTM(input_dim,1024)

        # extra layers layers
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.lstm(x)
        logits = self.fc2(x)

        return logits



params = {'batch_size': 100,
          'shuffle': True}
          #'num_workers':4}
model_params = {'input_dim': 4096,
                'hidden_dim': 2000,
                'output_dim': 2}

ffnet     = FeedForwardNetwork(**model_params)
lstmnet= LSTMNetwork(**model_params)
net=ffnet



#partition = {'train': train_IDs, 'test': test_IDs}

# Generators
training_set = Dataset(train_IDs, train_labels, train_values)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(test_IDs, test_labels,test_values)
validation_generator = data.DataLoader(validation_set, **params)




use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
print('device', device)
torch.backends.cudnn.benchmark = True

# Training
for local_batch, local_labels in validation_generator:
    # Transfer to GPU
    print(local_batch.shape)
    print(local_labels.shape)
    print(local_batch[0][:10])
    break

for local_batch, local_labels in training_generator:
    # Transfer to GPU
    print(local_batch.shape)
    print(local_labels.shape)
    #local_batch, local_labels = local_batch.to(device), local_labels.to(device)

    # Model computations
    # [...]
    break

# Validation
with torch.set_grad_enabled(False):
    for local_batch, local_labels in validation_generator:
        # Transfer to GPU
        print(local_batch.shape)
        print(local_labels.shape)
        #local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        # [...]
        break


def backprop(X, y, model, optimizer, loss_fn):
    # print(X[0][:10])
    # print(output.shape, y.shape)
    # print(output[0], y[0])

    output = model(X)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()

    return loss.item()

train_losses = {}
val_results  = {}
log_interval = 10
learning_rate = 0.001#3e-4
#weights = np.power(df.iloc[train_IDs][labels].value_counts().values, 2)
#weights = max(weights) * 1 / weights
#weights = torch.Tensor(weights)


loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,  momentum=0.9)



max_epochs = 100
for epoch in range(len(val_results), len(val_results) + max_epochs):
    # Training
    net.train()
    print('Epoch', epoch)
    train_losses[epoch] = []

    for batch, (local_batch, local_labels) in enumerate(tqdm(training_generator)):
        # tranfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        optimizer.zero_grad()
        l = backprop(local_batch, local_labels, net, optimizer, loss_fn)
        train_losses[epoch].append(l)

        if batch % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch * len(local_batch), len(training_generator.dataset),
                       100. * batch / len(training_generator), l))
    print(f'Average loss on epoch {epoch}: {np.mean(train_losses[epoch])}')

    # Validation
    net.eval()
    with torch.no_grad():
        val_results[epoch] = {'out': [], 'gt': []}
        for local_batch, local_labels in tqdm(validation_generator):
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            logits_output = net(local_batch)  # .float()
            # val_results[epoch]['out'].append(logits_output.cpu())
            # val_results[epoch]['gt'].append(local_labels.cpu())
            val_results[epoch]['out'].append(logits_output.to(device))
            val_results[epoch]['gt'].append(local_labels.to(device))

        all_out = [l for batch in val_results[epoch]['out'] for l in batch.numpy()]
        #print(all_out)
        all_gt = [l for batch in val_results[epoch]['gt'] for l in batch.numpy()]
        #print(all_gt)

        print('PRAUC:', compute_prauc(softmax(all_out)[:, 1], all_gt) * 100)
        print('RCE:', compute_rce(softmax(all_out)[:, 1], all_gt))
        print('APS:', average_precision_score(np.argmax(all_out, axis=1), all_gt))
        print('F1:',f1_score(np.argmax(all_out, axis=1), all_gt))


#print('PRAUC:', compute_prauc(softmax(all_out)[:, 1], all_gt) * 100)
#print('RCE:', compute_rce(softmax(all_out)[:, 1], all_gt))

#print(type(all_out))






#a=df_total['values'][3]
#print(a.dtype)
#df_total.to_csv('interestingness_2017.csv')
#print(df_total)
#video_77_2223-2251.ys



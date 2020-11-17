# BERT is a large scale transformer based language model that can be fine tuned for a variety of tasks

import torch
import pandas as pd
from tqdm.notebook import tqdm

import easygui
# /Users/anandsaimishra/Desktop/Master\'s Thesis/ProjectImplementation/Data/smile-annotations-final.csv

df = pd.read_csv(
    easygui.fileopenbox(),
    names=['id', 'text', 'category'])
'''
Our dataset is going to contain three values i.e. the id of the tweet, the tweet itself and the emtion that we are 
detecting that it represents. df is saving th data as a csv file.
'''
df.set_index('id', inplace=True)
print("\n______________________________\n")
'''We consider the id to be unique and hence we are deisgning it to be inplace and ordered'''
print("Exploratory Data Analysis: Let's peek into the given dataset")
print(df.head())
print("\n______________________________\n")
print("\nThe First tweet in this dataset is : ")
print(df.text.iloc[0])
'''
This next command shows us exactly how many times each emotion gets repeated.
'''
print("\n______________________________\n")
print("Lets see how many times each emotion is repeated in each of the tweets in the dataset \n")
print(df['category'].value_counts())
print("\n______________________________\n")
'''
The reason we have avoided the tweets with multiple emtions at a time is that it makes it very difficult to handle.

We need to remove all the nocode and multiple emotion catagories.

Hence we need to filter the unwanted catagories out
'''

df = df[~df['category'].str.contains('\|')]
df = df[df['category'] != 'nocode']

print('After Filtering the categories')
print(df['category'].value_counts())

'''
Now we need to build the dictionary which will link our emotions into a relavent number Key = Emotion & Value = Number
'''

possible_labels = df['category'].unique()
label_dict = {}
for index, possible_labels in enumerate(possible_labels):
    label_dict[possible_labels] = index


'''
What we now notice is that the dictionary is created
'''
print(label_dict)
'''
We use the replace function to replace catagory with the new dictionary function

'''
print("\n______________________________\n")
df['label'] = df['category'].replace(label_dict)
print(df.head())

print("\n______________________________\n")

'''
Creating the training and validation sets and createing the classes

We cant assume that all categories have the same variance and number of examples to be trained on the RNN. For ex  ample : 'Disgust'

Stratified Approach so we make the data split in a stratified fashion we use sklearn to train test split.

this will return four things x_train, x_val 
'''
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(df.index.values, df.label.values, test_size=0.15, random_state=17, stratify=df.label.values)

df['data_type'] = ['not_set']*df.shape[0]
print(df.head())
df.loc[x_train, 'data_type'] = 'train'
df.loc[x_val, 'data_type'] = 'val'
print(df.groupby(['category', 'label', 'data_type']).count())
'''
We now can confirm that each of our training set has been split according to the training and test sets.

Now we shall work on a tokenizer which will convert the text data into a numerical value.
We had to split the dataset into a training and validation datasets in a stratified manner and that 
is to preserve our representation of all of our classes in both the traininng and validation split.
'''
'''
Tokenizer - it takes a raw data and splits it into tokens. a token is simply a 
numerical representation of a perticular data.

'''
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
'''
We will be using pretrained to encode our dataset and we will be using bert-base-uncased uncased just means 
that we are using all uncased data as we convert everything to a lowercase.

'''
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].text.values,
    add_special_tokens=True, #
    return_attention_mask=True,
    pad_to_max_length = True,
    max_length=256,
    return_tensors = 'pt'
)
'''
add_special_tokens -> this is just BERT's way of knowing when a sentence ends and a new one begins
return_attention_mask-> Becouse we are using a fixed input we need all the data to be of the same dimentionality (256) 
this will show where the actual mask run and where is just zero
'''
encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].text.values,
    add_special_tokens=True, #
    return_attention_mask=True,
    pad_to_max_length = True,
    truncation=True,
    max_length=256,
    return_tensors = 'pt'
)
'''
We use almost an identical way to get the validation.
Then we need to split the dataset into input_ids, attention_maska and labels_traisn
Truncation to explicitely truncate examples to max length. Defaulting to 'longest_first' truncation strategy.
'''

input_ids_train = encoded_data_train['input_ids'] #This represents each word as a number essentially
attention_masks_train = encoded_data_train['attention_mask']#This is rather a PyTorch Tensor
label_train = torch.tensor(df[df.data_type == 'train'].label.values)

input_ids_val = encoded_data_val['input_ids'] #This represents each word as a number essentially
attention_masks_val = encoded_data_val['attention_mask']#This is rather a PyTorch Tensor
label_val = torch.tensor(df[df.data_type == 'val'].label.values)
'''
Now that we have our encoded data sets so we now convert it to two different datasets we will like to create 2
new datasets i.e dataset_train and dataset_val 
'''
dataset_train = TensorDataset(input_ids_train, attention_masks_train, label_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, label_val)

'''
We need to keep in chech the length of both our training and validation dataset.
'''
print("The Training dataset is of the length: " + str(len(dataset_train)))
print("The Validation dataset is of the length: " + str(len(dataset_val)))

'''
We encoded the text using the Bert Tokenizer.

Now we shall import our Pretrained Bert Model. We shall treat each single tweet as its own individual
sequence and it will be classified into one of the six classes. 

The reason we use BERT Base is becouse of the limited computations the larger general version of Bert
would be too large even just to infer and not even to actually train. And becouse we are using full sequence classification
this is our step to redefine the architecture to include

num label basically says how many output labels must this final Bert layer have.
We dont care about the feedback from the sets hence we shall put the output attentions to be off.
We dont care about the output of the last hidden state just before the output at all it is only helpful in the 
encoding process.
'''
from transformers import logging
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                      num_labels = len(label_dict),
                                      output_attentions = False,
                                      output_hidden_states=False
                                      )
'''
This will take a while to run and execute becouse it will have to download the baseBERT and it is a
large file (440MB)

Bert essentially takes in text and is able to encode the same in terms of the huge the data that it is initialy exposed to 
so we are adding a layer of data of size 6 and hence it becomes a classifier that selects the right emotion.
Now this is ready to be trained.

CREATING DATA LOADERS
the random and sequential sampler is to sample data from the dataset for analysis.
this helps in how we train the model and what data it is exposed to. we are randomly sorting the dataset for 
more variability.

'''
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 32
dataloader_train = DataLoader(
    dataset_train,
    sampler=RandomSampler(dataset_train),
    batch_size = batch_size
)
dataloader_val = DataLoader(
    dataset_train,
    sampler=RandomSampler(dataset_val),
    batch_size = batch_size
)
'''
We shall now introduce our Optimizer this just defines our learning rate and how it chages as we go through each epoc.
'''
from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(#Controls how learning rate changes through time
    model.parameters(),
    lr=1e-5,
    eps=1e-8
)
epochs = 10
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(dataloader_train)*epochs
)
'''
Performance Metrics we will use numpy and sklearn
'''
import numpy as np
from sklearn.metrics import f1_score


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted') # we use f1_score is becouse we know there is a class imbalance in our dataset.
def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')


'''
Creating the training Loop

'''
import random

seed_val = 17
random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)
#Generally used if we are using GPU

def evaluate(dataloader_val):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


for epoch in tqdm(range(1, epochs + 1)):

    model.train()

    loss_train_total = 0

    print("_________________________________________________")
    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:
        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

    torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model')
    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, predictions, true_vals = evaluate(dataloader_val) #This is important becouse you want to know if your model is over training.
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')

    '''
    Loading and evaluating our model
    Takes about 40 min per epoc. 
    
    We need to quit our model to the device
    
    '''
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=len(label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)

    model.to(device)
    model.load_state_dict(torch.load('/Users/anandsaimishra/Desktop/Master\'s Thesis/ProjectImplementation/Data/finetuned_bert_epoch_1_gpu_trained.model', map_location=torch.device('cpu')))
    _, predictions, true_vals = evaluate(dataloader_val)
    accuracy_per_class(predictions, true_vals)

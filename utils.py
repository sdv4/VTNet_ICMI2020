"""
This module contains miscellaneous functions required to run the Jupyter
notebooks associated with the "Predicting Confusion from Eye-Tracking Data with
Recurrent Neural Networks" experiments. The functions contained herein are
common to all notebooks, unless otherwise defined locally.
"""

import pickle
import random
import numpy as np
import pandas as pd

import torch
from torch import nn

from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.offsetbox import TextArea
from matplotlib.offsetbox import VPacker

MANUAL_SEED = 1
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

np.random.seed(MANUAL_SEED)
random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)


if DEVICE.type == 'cuda:3':
    torch.cuda.manual_seed(MANUAL_SEED)
    torch.cuda.manual_seed_all(MANUAL_SEED)
else:
    torch.manual_seed(MANUAL_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#global variables to be set upon import
MAX_SEQUENCE_LENGTH = -1
INPUT_SIZE = -1


def pickle_loader(input_file_path):
    """ Finishes the pre-processing of data items.

        Args:
            input_file_path (string): the path to the data item to be loaded
        Returns:
            item (numpy array): the fully processed data item

    """
    file = open(input_file_path, 'rb')
    item = pickle.load(file)
    item = item.values[-MAX_SEQUENCE_LENGTH:]
    if len(item) < MAX_SEQUENCE_LENGTH:
        num_zeros_to_pad = (MAX_SEQUENCE_LENGTH)-len(item)
        item = np.append(np.zeros((num_zeros_to_pad, INPUT_SIZE)), item, axis=0)
    file.close()
    return item

def pickle_loader2(input_file_path):
    """ Finishes the pre-processing of data items. Different from pickle_loader
        in that it makes negative gaze coordinates positive.

        Args:
            input_file_name (string): the name of the data item to be loaded
        Returns:
            item (numpy array): the fully processed data item

    """
    file = open(input_file_path, 'rb')
    item = pickle.load(file)
    item = item.values[-MAX_SEQUENCE_LENGTH:, 0:INPUT_SIZE]
    if INPUT_SIZE == 17:
        item[item[:,14] == 'KeyPress'] = 3

    if len(item) < MAX_SEQUENCE_LENGTH:
        num_zeros_to_pad = (MAX_SEQUENCE_LENGTH)-len(item)
        item = np.append(np.zeros((num_zeros_to_pad, INPUT_SIZE)), item, axis=0)
    file.close()
    item = abs(item)
    item[item == 1.0] = -1.0
    return item.astype('float64')

def pickle_loader3(input_file_path):
    """ Data loader for items not needing any further pre-processing. 

        Args:
            input_file_path (string): the path to the data item to be loaded
        Returns:
            item (numpy array): the fully processed data item

    """
    file = open(input_file_path, 'rb')
    item = pickle.load(file)
    file.close()
    item = np.array(item)
    return item

def pickle_loader4(input_file_path):
    """ 
        A loader that loads only the last 5 seconds worth of  fixations. If
        the end of a fixation is within the last 5 seconds, the entire fixation
        is loaded.
        
        Args:
            input_file_name (string): the name of the data item to be loaded
        Returns:
            item (numpy array): the fully processed data item

    """
    file = open(input_file_path, 'rb')
    item = pickle.load(file)
    
    accumulated_ms = 0
    i = 1
    while accumulated_ms < 5000 and i < (len(item)-1):
        accumulated_ms += item.iloc[-i,0]
        i+=1
    item = item.iloc[-(i-1):]
    item = item.values[-MAX_SEQUENCE_LENGTH:]
    if len(item) < MAX_SEQUENCE_LENGTH:
        num_zeros_to_pad = (MAX_SEQUENCE_LENGTH)-len(item)
        item = np.append(np.zeros((num_zeros_to_pad, INPUT_SIZE)), item, axis=0)
    file.close()
    return item


def setup_local_train_test_val(confused_path, not_confused_path,
                               batch_size, loader, 
                               train_items=None, test_items=None, val_items=None,
                               train_path=None, test_path=None, val_path=None, 
                               verbose=False):
    """
        Sets up datasets for training, testing, and validation data.
        Args:
            confused_path (string): relative path to confused items
            not_confused_path (string): relative path to not_confused items
            batch_size (int): size of batch to use for data loader objects
            loader (function): function to use to load data
            train_items (list,list): tuple containing list of confused and 
                not confused items in the training set, respectively.
            test_items (list,list): tuple containing list of confused and 
                not confused items in the test set, respectively.
            val_items (list,list): tuple containing list of confused and 
                not confused items in the validation set, respectively.
            train_path (string): a path where local training data can be
                stored without effecting other programs
            test_path (string): a path where local test data can be
                stored without effecting other programs
            val_path (string): a path where local val data can be
                stored without effecting other programs
            verbose (bool): if True, function prints the size of each dataset
    
        Raises:
            ValueError: raised if all of train_path, test_path, and val_path are None

        Returns:
            loaders (tuple): contains the following loaders if their corresponding
                input argument is not None:
                    train_loader (PyTorch DataLoader)
                    test_loader (PyTorch DataLoader)
                    val_loader (PyTorch DataLoader)
    """
    if train_path is None and test_path is None and val_path is None:
        raise ValueError("Function must return at least one data loader")

    if train_path is not None:
        local_train_confused_path = train_path + 'confused/'
        local_train_not_confused_path = train_path + 'not_confused/'
        train_confused_items = train_items[0]
        train_not_confused_items = train_items[1]
        
        if os.path.exists(local_train_confused_path):
            shutil.rmtree(local_train_confused_path)
        if os.path.exists(local_train_not_confused_path):
            shutil.rmtree(local_train_not_confused_path)
        
        os.makedirs(local_train_confused_path)
        for i in train_confused_items:
            shutil.copy(src=confused_path+i, dst=local_train_confused_path+i)
        
        os.makedirs(local_train_not_confused_path)
        for i in train_not_confused_items:
            shutil.copy(src=not_confused_path+i, dst=local_train_not_confused_path+i)
            
        training_data = datasets.DatasetFolder(train_path,
                                               loader=loader,
                                               extensions='.pkl')

        train_loader = torch.utils.data.DataLoader(training_data,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10 if \
                                                   DEVICE.type == \
                                                   'cuda' else 5,
                                                   pin_memory=True,
                                                   drop_last=False)
        if verbose:
            print("Training data: ", training_data)
        
    if test_path is not None:
        local_test_confused_path = test_path + 'confused/'
        local_test_not_confused_path = test_path + 'not_confused/'
        test_confused_items = test_items[0]
        test_not_confused_items = test_items[1]
    
        if os.path.exists(local_test_confused_path):
            shutil.rmtree(local_test_confused_path)
        if os.path.exists(local_test_not_confused_path):
            shutil.rmtree(local_test_not_confused_path)
        
        os.makedirs(local_test_confused_path)
        for i in test_confused_items:
            shutil.copy(src=confused_path+i, dst=local_test_confused_path+i)
        
        os.makedirs(local_test_not_confused_path)
        for i in test_not_confused_items:
            shutil.copy(src=not_confused_path+i, dst=local_test_not_confused_path+i)
            
        test_data = datasets.DatasetFolder(test_path,
                                           loader=loader,
                                           extensions='.pkl')

        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=10 if \
                                                  DEVICE.type == \
                                                  'cuda' else 5,
                                                  pin_memory=True,
                                                  drop_last=False)
        if verbose:
            print("Test data: ", test_data)
            
    if val_path is not None:
        local_val_confused_path = val_path + 'confused/'
        local_val_not_confused_path = val_path + 'not_confused/'
        val_confused_items = val_items[0]
        val_not_confused_items = val_items[1]
    
        if os.path.exists(local_val_confused_path):
            shutil.rmtree(local_val_confused_path)
        if os.path.exists(local_val_not_confused_path):
            shutil.rmtree(local_val_not_confused_path)
        
        os.makedirs(local_val_confused_path)
        for i in val_confused_items:
            shutil.copy(src=confused_path+i, dst=local_val_confused_path+i)
        
        os.makedirs(local_val_not_confused_path)
        for i in val_not_confused_items:
            shutil.copy(src=not_confused_path+i, dst=local_val_not_confused_path+i)
            
        val_data = datasets.DatasetFolder(val_path,
                                          loader=loader,
                                          extensions='.pkl')

        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=10 if \
                                                 DEVICE.type == \
                                                 'cuda' else 5,
                                                 pin_memory=True,
                                                 drop_last=False)
        if verbose:
            print("Validation data: ", val_data)
        
    loaders = []
    if train_path is not None:
        loaders.append(train_loader)
    if test_path is not None:
        loaders.append(test_loader)
    if val_path is not None:
        loaders.append(val_loader)
        
    return loaders


def get_min_max_feature_values(path_confused, path_not_confused, confused_items,
                               not_confused_items):
    """ Function to get the min and max values for each feature in a dataset.

        Args:
            path_confused (string): path to the confuued items of the dataset
            path_not_confused (string): path to not_confused items of the dataset
            confused_items (list): file names of confused in dataset
            not_confused_items (list): file names of not_confused in dataset

        Returns:
            min_feature_values (list): each element is the min value for the
                corresponding feature in the dataset
            max_feature_values (list): each element is the max value for the
                corresponding feature in the dataset
    """

    all_training_data = 0
    for i, item in enumerate(confused_items):
        if i == 0:
            all_training_data = pickle_loader(path_confused+item)
        else:
            all_training_data = np.concatenate((all_training_data,
                                                pickle_loader(path_confused+item)))

    for i, item in enumerate(not_confused_items):
        all_training_data = np.concatenate((all_training_data,
                                            pickle_loader(path_not_confused+item)))


    max_feature_values = np.max(all_training_data, axis=0)
    all_training_data[all_training_data <= 0] = 1000
    min_feature_values = np.min(all_training_data, axis=0)

    return min_feature_values, max_feature_values


def normalize_inputs(inputs, min_feature_values, max_feature_values):
    """ Takes a batch of input and returns the min-maxed normalized version.

        Args:
            inputs (PyTorch Tensor): batch of input values of shape
                (batch_size,MAX_SEQUENCE_LENGTH,INPUT_SIZE)
            min_feature_values (np array): array of shape (INPUT_SIZE, ), where
                each element is the min value of the feature at that index
            max_feature_values (np array): array of shape (INPUT_SIZE, ), where
                each element is the max value of the feature at that index

        Returns:
            normalize_inputs (np array): the normalized version of inputs
    """
    for i in range(len(min_feature_values)):
        if min_feature_values[i] == 0.0:
            min_feature_values[i] = 0.00000000001
    # duplicate min and max _feature_values for each row of a given item:
    min_reshaped = np.array([min_feature_values,]*inputs.shape[1])
    max_reshaped = np.array([max_feature_values,]*inputs.shape[1])

    # duplicate the duplicated min and max _feature_values for item in batch:
    min_reshaped = np.array([min_reshaped]*inputs.shape[0])
    max_reshaped = np.array([max_reshaped]*inputs.shape[0])

    # normalize x as x = x-min/(max-min)
    max_minus_min = max_reshaped - min_reshaped
    max_minus_min[max_minus_min == 0.0] = 0.00000000001

    normalized_inputs = (inputs - min_reshaped) / (max_minus_min)
    normalized_inputs[normalized_inputs > 1.0] = 1.0
    return normalized_inputs


def get_mean_std_feature_values(path_confused, path_not_confused, confused_items,
                                not_confused_items):
    """ Function to get the mean and standard deviation values for each feature
        in a dataset.

        Args:
            path_confused (string): path to the confuued items of the dataset
            path_not_confused (string): path to not_confused items of the dataset
            confused_items (list): file names of confused in dataset
            not_confused_items (list): file names of not_confused in dataset

        Returns:
            mean_feature_values (list): each element is the mean value for the
                corresponding feature in the dataset
            std_feature_values (list): each element is the standard deviation
                value for the corresponding feature in the dataset
    """

    all_training_data = 0
    for i, item in enumerate(confused_items):
        if i == 0:
            all_training_data = pickle_loader(path_confused+item)
        else:
            all_training_data = np.concatenate((all_training_data,
                                                pickle_loader(path_confused+item)))

    for i, item in enumerate(not_confused_items):
        all_training_data = np.concatenate((all_training_data,
                                            pickle_loader(path_not_confused+item)))


    mean_feature_values = np.mean(all_training_data[all_training_data != -1.0],
                                  axis=0)

    std_feature_values = np.std(all_training_data[all_training_data != -1.0],
                                axis=0)

    return mean_feature_values, std_feature_values

def standardize_inputs(inputs, mean_feature_values, std_feature_values):
    """ Takes a batch of input and returns the standardized version, such that
        the mean value of each feature is 0 and standard deviation is 1.

        Args:
            inputs (PyTorch Tensor): batch of input values of shape
                (batch_size,MAX_SEQUENCE_LENGTH,INPUT_SIZE)
            mean_feature_values (np array): array of shape (INPUT_SIZE, ), where
                each element is the mean value of the feature at that index
            std_feature_values (np array): array of shape (INPUT_SIZE, ), where
                each element is the standard deviation of the feature at that index

        Returns:
            standardized_inputs (np array): the standardized version of inputs
    """
    # duplicate mean and std _feature_values for each row of a given item:
    mean_reshaped = np.array([mean_feature_values,]*inputs.shape[1])
    std_reshaped = np.array([std_feature_values,]*inputs.shape[1])

    # duplicate the duplicated min and max _feature_values for item in batch:
    mean_reshaped = np.array([mean_reshaped]*inputs.shape[0])
    std_reshaped = np.array([std_reshaped]*inputs.shape[0])

    # standardize as (x-mean) / standard deviatoin
    standardized_inputs = (inputs - mean_reshaped)/ std_reshaped

    return standardized_inputs


def check_metrics(model,
                  data_loader,
                  verbose=False,
                  threshold=None,
                  return_threshold=False,
                  normalize=False,
                  min_feat_values=None,
                  max_feat_values=None):
    """ Computes the model accuracy, recall, specificity, ROC, FP rate, TP
        rate, thresholds for ROC curve on a given dataset.

    Args:
        model (PyTorch model): model whose accuracy will be tested
        data_loader (PyTorch DataLoader): the dataset over which metric are calculted
        verbose (Boolean): if True will print the accuracy as a %, size of the
            dataset, recall, and specificity
        threshold (float): if given, this threshold is used to calculate the metrics
        return_threshold (boolean): if True, the calculated optimal threshold is returned
            
    Returns:
        (float, float, float, float, float): if return_threshold is False: accruacy, recall,
            specificity, AUC, and NLLLoss. All in [0.0,1.0].
        (float, float, float, float, float, float): if return_threshold is True: accruacy,
            recall, specificity, AUC, NLLLoss, and threshold. All in [0.0,1.0].
    """
    correct = 0
    total = 0
    y_true = np.array([])
    y_pred = np.array([])
    y_0_scores = np.array([])

    # Make predictions
    with torch.no_grad():
        criterion = nn.NLLLoss()
        criterion = criterion.to(DEVICE)
        loss = 0
        model = model.eval()
        batches = 1
        for i, data in enumerate(data_loader, 1):
            # get the inputs
            inputs, labels = data
            model_arch = model.get_architecture()
            if model_arch == 'gru' or model_arch == 'lstm':
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                hidden = model.init_hidden(batch_size=labels.shape[0])
                if len(hidden) == 2:
                    if len(hidden) == 2:
                        hidden = (hidden[0].to(DEVICE), hidden[1].to(DEVICE))
                else:
                    hidden = hidden.to(DEVICE)
                for j in range(MAX_SEQUENCE_LENGTH):
                    outputs, hidden = model(inputs[:, j].unsqueeze(1).float(), hidden)
            else:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs.float())

            y_true = np.concatenate((labels.cpu().numpy(), y_true), axis=None)
            loss += (criterion(outputs, labels.squeeze(0).long()).item())

            total += labels.size(0)
            y_0_scores = np.concatenate((torch.exp(outputs).cpu().numpy()[:, 0],
                                         y_0_scores), axis=0)
            batches = i
        # Compute metrics
        loss = loss / batches

        # no option to use 0 (confused) as positive class label,
        # so flip lables so that the true positive class (0) is
        # represented with 1s:
        y_true_flipped = y_true.copy()
        y_true_flipped[y_true == 1] = 0
        y_true_flipped[y_true == 0] = 1
        auc = roc_auc_score(y_true_flipped, y_0_scores)

        # roc_curve expects y_scores to be probability values of the positive class
        fpr, tpr, thresholds = roc_curve(y_true, y_0_scores, pos_label=0)
        if threshold is not None:
            print("Calculating metrics with given threshold")
            y_pred = y_0_scores < threshold # use < so that neg class maintains the 1 label
            correct = sum(y_pred == y_true)
            accuracy = correct/len(y_pred)
            recall = recall_score(y_true, y_pred, pos_label=0)
            specificity = recall_score(y_true, y_pred, pos_label=1)
        else:
            if return_threshold:
                recall, specificity, accuracy, \
                opt_thresh = optimal_threshold_sensitivity_specificity(thresholds[1:],
                                                                       tpr[1:],
                                                                       fpr[1:],
                                                                       y_true,
                                                                       y_0_scores,
                                                                       True)
            else:
                recall, specificity, \
                accuracy = optimal_threshold_sensitivity_specificity(thresholds[1:],
                                                                     tpr[1:],
                                                                     fpr[1:],
                                                                     y_true,
                                                                     y_0_scores)
        if verbose:
            print('Accuracy of the network on the ' + str(total) +
                  ' data items: %f %%' % (correct / total))
            print("Loss: ", loss)
            print("Recall/Sensitivity: ", recall)
            print("Specificity: ", specificity)
            print("AUC: ", auc)

    model = model.train()
    if return_threshold:
        metrics = (accuracy, recall, specificity, auc, loss, opt_thresh)
    else:
        metrics = (accuracy, recall, specificity, auc, loss)

    return metrics


def optimal_threshold_sensitivity_specificity(thresholds,
                                              true_pos_rates,
                                              false_pos_rates,
                                              y_true,
                                              y_0_scores,
                                              return_thresh=False):
    """ Finds the optimal threshold then calculates sensitivity and specificity.

    Args:
        thresholds (list): the list of thresholds used in computing the ROC score.
        true_pos_rates (list): the TP rate corresponding to each thresholds.
        false_pos_rates (list): the FP rate corresponding to each threshold.
        y_true (list): the ground truth labels of the dataset over which
            sensitivity and specificity will be calculated.
        y_0_scores (list): the model's probability that each item in the dataset is
            class 0, (i.e. confused).
        return_thresh (boolean): if True, the calculated optimal threshold is returned

    Returns:
        sensitivity (float): True positive rate when optimal threshold is used
        specificity (float): True negative rate when optimal threshold is used
        accuracy (float): the percentage of lables that were correctly predicted, in [0,1]
        best_threshold (float): if return_thresh is true, this value is the
            decition threshold that maximized combined sensitivity and specificity
    """

    best_threshold = 0.5
    dist = -1
    for i, thresh in enumerate(thresholds):
        current_dist = np.sqrt((np.power(1-true_pos_rates[i], 2)) +
                               (np.power(false_pos_rates[i], 2)))
        if dist == -1 or current_dist <= dist:
            dist = current_dist
            best_threshold = thresh

    y_pred = (y_0_scores >= best_threshold) == False
    y_pred = np.array(y_pred, dtype=np.int)
    #accuracy = sum(y_pred == y_true)/len(y_true)
    accuracy = -1    
    sensitivity = recall_score(y_true, y_pred, pos_label=0)
    specificity = recall_score(y_true, y_pred, pos_label=1)

    if return_thresh:
        metrics = (sensitivity, specificity, accuracy, best_threshold)
    else:
        metrics = (sensitivity, specificity, accuracy)

    return metrics


def batch_accuracy(predictions, ground_truth):
    """ Calculate accuracy of predictions over items in a single batch.

    Args:
        predictions (PyTorch Tensor): the logit output of datum in the batch
        ground_truth (PyTorch): the correct class index of each datum

    Returns
        (float): the % of correct predictions as a value in [0,1]
    """

    correct_predictions = torch.argmax(predictions, dim=1) == ground_truth

    return torch.sum(correct_predictions).item()/len(correct_predictions)



def get_grouped_splits(confused_items, not_confused_items, k):
    """ Splits data ensuring no users have data in training and eval sets.

        Args:
            confused (list): list of data item names labelled as confused
            not_confused (list): list of data item names labelled as not_confused
            k (int): number of folds for cross validation.

        Returns: (in following order)
            train_confused_splits (list): each element is a list containing the
                file names of the data items for this partition of the dataset
            test_confused_splits (list): as above
            train_not_confused_splits (list): as above
            test_not_confused_splits (list): as above
    """

    train_confused_splits = []
    test_confused_splits = []
    train_not_confused_splits = []
    test_not_confused_splits = []

    # make list where each index corresponds to the "group" (userID)
    groups = [uid.split('_')[0][:-1] for uid in confused_items] + \
             [uid.split('_')[0][:-1] for uid in not_confused_items]
    # get train test splits for confused class
    dummy_y = [0 for i in range(len(confused_items))] + \
              [1 for i in range(len(not_confused_items))]

    items = confused_items + not_confused_items

    gkf = GroupKFold(n_splits=k)
    gkf.get_n_splits(X=items, y=dummy_y, groups=groups)
    for train, test in gkf.split(X=items, y=dummy_y, groups=groups):
        train_confused_splits.append([items[i] for i in train if dummy_y[i] == 0])
        test_confused_splits.append([items[i] for i in test if dummy_y[i] == 0])
        train_not_confused_splits.append([items[i] for i in train if dummy_y[i] == 1])
        test_not_confused_splits.append([items[i] for i in test if dummy_y[i] == 1])

    return (train_confused_splits, test_confused_splits,
            train_not_confused_splits, test_not_confused_splits)



def get_train_val_split(confused_items, not_confused_items, percent_val_set):
    """ Grouped split the training set into a training and validation set.

        Args:
            confused_items (list): list of strings; each of which is the name of
                a file containing a data item labelled confused.
            not_confused_items (list): list of strings; each of which is the
                name of a file containing a data item labelled not_confused.

        Returns:
            train_confused (list): list of strings; each is the name of a data
                item in the training set.
            train_not_confused (list): list of strings; each is the name of a
                data item in the training set.
            val_confused (list): list of strings; each is the name of a data
                item in the training set.
            val_not_confused (list): list of strings; each is the name of a
                data item in the training set.
    """

    train_confused = []
    val_confused = []
    train_not_confused = []
    val_not_confused = []

    # make list where each index corresponds to the "group" (userID)
    confused_groups = [uid.split('_')[0][:-1] for uid in confused_items]
    not_confused_groups = [uid.split('_')[0][:-1] for uid in not_confused_items]
    groups = confused_groups + not_confused_groups
    dummy_y = [0 for i in range(len(confused_items))] + \
              [1 for i in range(len(not_confused_items))]
    items = confused_items + not_confused_items
    gkf = GroupShuffleSplit(n_splits=1, test_size=percent_val_set,
                            random_state=MANUAL_SEED)
    gkf.get_n_splits(X=items, y=dummy_y, groups=groups)

    for train, test in gkf.split(X=items, y=dummy_y, groups=groups):


        train_confused = [items[i] for i in train if dummy_y[i] == 0]
        train_not_confused = [items[i] for i in train if dummy_y[i] == 1]

        val_confused = [items[i] for i in test if dummy_y[i] == 0]
        val_not_confused = [items[i] for i in test if dummy_y[i] == 1]

    return train_confused, train_not_confused, val_confused, val_not_confused


def _plot_train_val_loss(training_losses, validation_losses):
    """ Plots training and validation loss in same figure.

        Args:
            training_losses (list)
            validation_losses (list)
    """
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6)
    num_training_losses = np.arange(0, len(training_losses), 1)
    color = 'tab:red'
    ax1.set_xlabel('Training Iterations')
    ax1.set_ylabel('Training Loss (%)', color=color)
    ax1.plot(num_training_losses, training_losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0.0, 1.1])

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Validation Loss', color=color)
    ax2.plot(num_training_losses, validation_losses, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.0, 1.1])
    fig.tight_layout()
    plt.show()


def _plot_train_val_acc(training_accs, validation_accs):
    """ Plots training and validation accuracy in same figure.

        Args:
            training_accs (list)
            validation_accs (list)
    """
    fig, ax1 = plt.subplots()
    ax1.set_ylim([0.1, 1.0])
    fig.set_size_inches(12, 6)
    num_val_accs = np.arange(0, len(validation_accs), 1)
    color = 'tab:blue'

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation accuracy (%)', color=color)
    ax1.plot(num_val_accs, validation_accs, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0.0, 1.1])

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Training accuracy (%)', color=color)
    ax2.plot(num_val_accs, training_accs, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.0, 1.1])

    fig.tight_layout()
    plt.show()

def _plot_train_val_roc(training_aucs, validation_aucs):
    """ Plots training and validation auc roc score in same figure.

        Args:
            training_aucs (list)
            validation_aucs (list)
    """
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6)
    num_val_aucs = np.arange(0, len(validation_aucs), 1)
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation AUC', color=color)
    ax1.plot(num_val_aucs, validation_aucs, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0.0, 1.1])

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Training AUC', color=color)
    ax2.plot(num_val_aucs, training_aucs, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.0, 1.1])
    fig.tight_layout()
    plt.show()

def _plot_val_sens_specif_auc(validation_aucs,
                              validation_recalls,
                              validation_specifs):
    """ Plots training and validation auc roc score in same figure.

        Args:
            validation_aucs (list)
            validation_recalls (list)
            validation_specifs (list)
    """
    color = 'tab:red'
    num_val_aucs = np.arange(0, len(validation_aucs), 1)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 6)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation AUC', color=color)
    ax1.plot(num_val_aucs, validation_aucs, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0.0, 1.1])

    ax2 = ax1.twinx()
    ax2.set_ylim([0.0, 1.1])


    ybox1 = TextArea("Sensitivity ", textprops=dict(color='tab:blue',
                                                    rotation=90, ha='left',
                                                    va='bottom'))
    ybox2 = TextArea("and ", textprops=dict(color="k", rotation=90, ha='left',
                                            va='bottom'))
    ybox3 = TextArea("Specificity ", textprops=dict(color='xkcd:azure',
                                                    rotation=90, ha='left',
                                                    va='bottom'))

    ybox = VPacker(children=[ybox1, ybox2, ybox3], align="bottom", pad=0, sep=5)

    anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False,
                                      bbox_to_anchor=(1.13, 0.25),
                                      bbox_transform=ax2.transAxes, borderpad=0.)

    ax2.add_artist(anchored_ybox)

    color = 'tab:blue'
    ax2.plot(num_val_aucs, validation_recalls, color=color)
    ax2.plot(num_val_aucs, validation_specifs, color='xkcd:azure')

    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()


def plot_metrics(training_accs,
                 training_losses,
                 training_aucs,
                 validation_accs,
                 validation_losses,
                 validation_recalls,
                 validation_specifs,
                 validation_aucs):
    """
        Outputs four graphs showing changes in metric values over training.
        1. Training and validation loss
        2. Training and validation accuracy
        3. Training and validation AUC ROC score
        4. Validation AUC ROC, sensitivity, and specificity
    """

    _plot_train_val_loss(training_losses, validation_losses)
    _plot_train_val_acc(training_accs, validation_accs)
    _plot_train_val_roc(training_aucs, validation_aucs)
    _plot_val_sens_specif_auc(validation_aucs,
                              validation_recalls,
                              validation_specifs)


def get_users(file_names):
    """ Returns the users whose items make up a given list of data items.

        Args:
            file_names (list): list of strings naming data items

        Returns:
            users (list): list of strings where each is a user whose data
                is in file_names
    """

    users = []
    for item in file_names:
        user_number = item.split('_')[0][:-1]
        if user_number not in users:
            users.append(user_number)
    return users

def batch_accuracy(predictions, ground_truth):
    """ Calculate accuracy of predictions over items in a single batch.
    Args:
        predictions (PyTorch Tensor): the logit output of datum in the batch
        ground_truth (PyTorch): the correct class index of each datum
    Returns
        (float): the % of correct predictions as a value in [0,1]
    """

    correct_predictions = torch.argmax(predictions, dim=1) == ground_truth

    return torch.sum(correct_predictions).item()/len(correct_predictions)
"""
This module contains the training functions required to train the RNN models in
the jupyter notebooks associated with the "Predicting Confusion from
Eye-Tracking Data with Recurrent Neural Networks" experiments. The functions
herein are common to all experiments.
"""

import random
import numpy as np

import torch
from torch import nn
from torch import optim

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import utils

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

#global variables set upon import
MAX_SEQUENCE_LENGTH = -1
INPUT_SIZE = -1
BATCH_SIZE = -1

utils.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
utils.INPUT_SIZE = INPUT_SIZE

def plot_grad_flow2(named_parameters):
    """Plots the gradients flowing through model during training.

        Usage: after loss.backwards()
        "plot_grad_flow(self.model.named_parameters())" to visualize the
        gradient flow.

        source: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8

        Args:
            named_parameters (PyTorch Parameters): named parameters to be visualized
    """

    ave_grads = []
    layers = []
    for param_name, param in named_parameters:
        if(param.requires_grad) and ("bias" not in param_name):
            layers.append(param_name)
            ave_grads.append(param.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
    
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
        source: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8

    """
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    
def jitter_params(model, params_to_jitter='all_params', scale_term=0.1):
    """ Add Gaussian noise (jitter) to model parameters.

        Args:
            model (PyTorch model): the models whose parameters will be jittered
            params_to_jitter (string): the name of the named parameters to apply
                noise to. Either 'all_params','input_params', 'weights_only'
            scale_term (float): amount to scale noise term by

        Returns:
            model (PyTorch model): model with jittered parameters
    """
    with torch.no_grad():
        for param in model.named_parameters():
            name = param[0]
            parameters = param[1]
            jitter = torch.randn(parameters.size())
            if params_to_jitter == 'all_params':
                parameters.add_((jitter * scale_term).to(DEVICE))

            elif params_to_jitter == 'weights_only':
                if 'bias' not in name:
                    parameters.add_((jitter * scale_term).to(DEVICE))

            elif params_to_jitter == 'input_params':
                if 'ih' in name:
                    parameters.add_((jitter * scale_term).to(DEVICE))

            else:
                raise NotImplementedError

    return model


def jitter_inputs(inputs, variance=1.0, scale_term=1):
    """ Adds normally distributed noise/jitter input values.

        Args:
            inputs (PyTorch Tensor): a batch of inputs with
                shape (BATCH_SIZE, MAX_SEQUENCE_LENGTH, INPUT_SIZE)
            variance (float): the variance of the sampling distribution
            scale_term (float): factor to scale sampled values by.

        Returns:
            jittered_inputs (PyTorch Tensor): inputs with added noise.

    """
    # mask invalid values (-1) and values that can only be 0 or 4:
    #inputs2 = inputs.data.numpy()
    inputs_mask = ((inputs != 0.0) * (inputs != -1.0) *
                   (inputs != 4.0)).float()

    dist = torch.distributions.normal.Normal(torch.Tensor([0.0]),
                                             torch.Tensor([variance]))

    jitter = (torch.Tensor(inputs_mask) *
              dist.sample(inputs_mask.shape).squeeze()) * scale_term
    jittered_inputs = inputs + (jitter.double())

    return jittered_inputs

def jitter_inputs_with_single_val(inputs, variance=1.0, scale_term=1):
    """ Adds the same noise normally distributed noise/jitter term to all
        features of input values.

        Args:
            inputs (PyTorch Tensor): a batch of inputs with
                shape (BATCH_SIZE, MAX_SEQUENCE_LENGTH, INPUT_SIZE)
            variance (float): the variance of the sampling distribution
            scale_term (float): factor to scale sampled values by.

        Returns:
            jittered_inputs (PyTorch Tensor): inputs with added noise.

    """
    # mask invalid values (-1) and values that can only be 0 or 4:
    inputs2 = inputs.data.numpy()
    inputs_mask = ((inputs2 != 0.0) * (inputs != -1.0) *
                   (inputs != 4.0)).float()

    dist = torch.distributions.normal.Normal(torch.Tensor([0.0]),
                                             torch.Tensor([variance]))
    jitter_term = dist.sample()[0].item() * scale_term
    jitter = (torch.Tensor(inputs_mask) *
              torch.Tensor(np.full(inputs_mask.shape, jitter_term)))
    jittered_inputs = inputs + (jitter.double())

    return jittered_inputs


def get_all_data(dataset, num_workers=30, shuffle=False):
    """ Loads all data from a given dataset into a list of batches
        that exists on the GPU.
        source: https://discuss.pytorch.org/t/how-to-collect-all-data-from-dataloader/15852
    """
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                              num_workers=num_workers, shuffle=shuffle,
                                              drop_last=False)
    all_data = []
    for sample_batched in data_loader:
        all_data.append(sample_batched)
    return all_data

def get_all_data2(dataset, num_workers=30, shuffle=False):
    """ Loads all data from a given dataset into a dicitonary that exists
        on the GPU.
        source: https://discuss.pytorch.org/t/how-to-collect-all-data-from-dataloader/15852
    """
    dataset_size = len(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_size,
                                        num_workers=num_workers, shuffle=shuffle,
                                        drop_last=False)
    all_data = {}
    for i_batch, sample_batched in enumerate(data_loader):
        all_data = sample_batched
    return all_data

def train(model,
          epochs,
          criterion_type,
          optimizer_type,
          training_data,
          val_data,
          print_every,
          plot_every,
          input_type='sequences',
          early_stopping=False,
          early_stopping_metric='val_auc',
          early_stopping_patience=4,
          rate_decay_patience=5,
          max_rate_decreases=0,
          initial_learning_rate=0.001,
          reload_best_on_decay=True,
          model_name='best_rnn',
          min_auc=0.0,
          add_noise=None,
          noise_variance=1.0,
          noise_scale_term=1.0,
          verbose=False,
          return_thresh=False):

    """
        Trains the given model according to the input arguments.

    Args:
        models (PyTorch model): The RNN to be trained.
        epochs (int): max number of iterations over the training dataset
        criterion (PyTorch loss): the loss function to use while backpropagating
            to train model
        optimizer (PyTorch optimizer): the optimization algorithm to be used
            train parameters of model
        training_data (PyTorch DatasetFolder): the training dataset that will be
            iterated through
        val_data (PyTorch DatasetFolder): the validation dataset that will be
            used for testing generalization
        print_every (int): the number of batches to pass between printing
            average loss and accuracy per batch for the batches since the last print
        plot_every (int): the number of batches to pass between recording metrics
            for plotting
        input_type (string): the form that the input items will take
        early_stopping (boolean): stop when validation metric of choice doesn't
            improve for patience epochs
        early_stopping_metric (string): the validation set metric to be
            monitored. Can be either 'val_auc' or 'val_loss'.
        early_stopping_patience (int): number of epochs with no validation
            accuracy imporvement to do before stopping
        rate_decay_patience (int): number of epochs without improvement to pass
            before reducing learning rate, when max_rate_decreases > 0
        max_rate_decreases (int): max number of times to decay learning rate
        initial_learning_rate (float): the first and max learning rate used
            by the optimizer
        model_name (string): the name with which to save the model to the
            working directory, which will be model_name.pt
        min_auc (float): the min. auc over validation set that must be achieved
            before saving the model.
        add_noise (string): if not None, then one of 'input_values', 'input_params',
            'all_params', or 'weights_only'
        noise_variance (float): the variance of the sampling distribution
        noise_scale_term (float): amount to scale noise
        verbose (boolean): print metrics each epoch if True.
        val_thresh (boolean): when True, the function returns the optimal
            threshold used to compute metrics over the validation set.

    Raises:
        ValueError: when input_type not one of 'sequences' or 'hidden_states'. If 
            'hidden_states' then model_type must also be LSTM
    
    Returns:

        if return_thresh:
             model (Pytorch model): the trained RNN
             training_accs (list): training set accuracy for each fold of CV
             validation_accs (list): validation set accuracy for each fold of CV
             training_losses (list): training set loss for each fold of CV
             training_aucs (list): training set auc roc score for each fold of CV
             validation_losses (list): validation set loss for each fold of CV
             validation_recalls (list): validation set recall score for each fold of CV
             validation_specifs (list): validation set specificity for each fold of CV
             validation_aucs (list): validation set auc roc score for each fold of CV
             val_thresh (float): the optimal threshold for computing sensitivity and
                 specificity as computed on the validation set

        else val_thresh is not returned

    """
    if input_type not in ('sequences', 'hidden_states'):
        raise ValueError("Invalid input_form: ", input_type)        
    
    model = model.train()

    best_auc = min_auc
    best_val_loss = 10000
    epochs_no_improvement = 0
    num_rate_decreases = 0

    # hold metrics to be tracked across training
    training_losses = []
    validation_recalls = []
    training_accs = []
    training_aucs = []
    validation_accs = []
    validation_losses = []
    validation_specifs = []
    validation_aucs = []
    
    # load all validation data onto the GPU
    val_loader = get_all_data(val_data, shuffle=False)
    # Check and record untrained validation metrics
    val_acc, \
    val_recall, \
    val_specif, \
    auc, \
    val_loss = utils.check_metrics(model, val_loader, verbose=False)

    validation_accs.append(val_acc)
    validation_losses.append(val_loss)
    validation_recalls.append(val_recall)
    validation_specifs.append(val_specif)
    validation_aucs.append(auc)
    
    #load all training data onto the GPU
    train_loader = get_all_data(training_data, shuffle=True)
    # Check and record untrained training set metrics
    training_acc, \
    _, \
    _, \
    training_auc, \
    training_loss = utils.check_metrics(model, train_loader, verbose=False)

    training_accs.append(training_acc)
    training_losses.append(training_loss)
    training_aucs.append(training_auc)

    if verbose:
        print("METRICS OF UNTRAINED MODEL")
        print("validation accuracy: ", val_acc)
        print("validation loss: ", val_loss)
        print("validation recall: ", val_recall)
        print("validation specificity: ", val_specif)
        print("validation AUC:, ", auc)
        print("training AUC: ", training_auc)

    learning_rate = initial_learning_rate
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    else:
        raise ValueError("ERROR: optimizer " + str(optimizer_type) +
                         " not supported.")
    if criterion_type == 'NLLLoss':
        criterion = nn.NLLLoss()
        criterion = criterion.to(DEVICE)
    else:
        raise ValueError("ERROR: criterion " + str(criterion_type) +
                         " not supported.")
    
    train_loader = get_all_data(training_data, shuffle=True)

    for epoch in range(epochs):
        
        # batch-wise metrics to be tracked
        training_accuracy = 0.0 # for printing accuracy over training set over epoch w/o recomputing
        running_acc = 0.0
        running_loss = 0.0
        plot_running_acc = 0.0
        plot_running_loss = 0.0
        num_batches = 0
        torch.manual_seed(MANUAL_SEED)
        for i, data in enumerate(train_loader, 0):

            # get the inputs
            inputs, labels = data
            model_arch = model.get_architecture()
            if model_arch == 'gru' or model_arch == 'lstm':
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                hidden = model.init_hidden(batch_size=labels.shape[0])
                if len(hidden) == 2:
                    hidden = (hidden[0].to(DEVICE), hidden[1].to(DEVICE))
                else:
                    hidden = hidden.to(DEVICE)
                for j in range(MAX_SEQUENCE_LENGTH):
                    outputs, hidden = model(inputs[:, j].unsqueeze(1).float(), hidden)
            else:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs.float())
                
            loss = criterion(outputs, labels.squeeze(0).long()).sum()

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if verbose:
                plot_grad_flow(model.named_parameters())

            # Update tracked metrics for batch
            batch_acc = utils.batch_accuracy(outputs, labels)
            training_accuracy += batch_acc
            running_loss += loss.item()
            running_acc += batch_acc
            plot_running_loss += loss.item()
            plot_running_acc += batch_acc

            if i % print_every == (print_every-1):
                print('[epoch: %d, batches: %5d] loss: %.5f | accuracy: %.5f'
                      % (epoch + 1, i + 1, running_loss/print_every,
                         running_acc/print_every))

                running_loss = 0.0
                running_acc = 0.0

            if i % plot_every == (plot_every-1):
                training_accs.append(plot_running_acc/plot_every)
                training_losses.append(plot_running_loss/plot_every)

                plot_running_loss = 0.0
                plot_running_acc = 0.0
                model = model.eval()

                val_acc, \
                val_recall, \
                val_specif, \
                auc, \
                val_loss = utils.check_metrics(model, val_loader, verbose=False)


                validation_accs.append(val_acc)
                validation_losses.append(val_loss)
                validation_recalls.append(val_recall)
                validation_specifs.append(val_specif)
                validation_aucs.append(auc)
                
                model = model.train()

                training_acc, \
                _, \
                _, \
                training_auc, \
                training_loss = utils.check_metrics(model, train_loader, verbose=False)
                training_aucs.append(training_auc)

            num_batches += 1

        # Update tracked metrics for epoch: accuracy, recall, specificity, auc, loss
        model = model.eval()
        val_acc, \
        val_recall, \
        val_specif, \
        val_auc, \
        val_loss = utils.check_metrics(model, val_loader, verbose=False)
        model = model.train()

        train_acc = training_accuracy/num_batches
        
        if verbose:
            print("Training accuracy for epoch: ", train_acc)
            print("validation accuracy: ", val_acc)
            print("validation loss: ", val_loss)
            print("validation recall: ", val_recall)
            print("validation specificity: ", val_specif)
            print("validation AUC: ", val_auc)

        if early_stopping:
            if early_stopping_metric == 'val_loss' and val_loss < best_val_loss:
                print("Old best val_loss: ", best_val_loss)
                print("New best val_loss: ", val_loss)
                print("Validation AUC: ", val_auc)
                best_val_loss = val_loss
                torch.save(model.state_dict(), './'+ model_name +'.pt')
                print("New best model found. Saving now.")
                epochs_no_improvement = 0
            elif early_stopping_metric == 'val_auc' and val_auc > best_auc:
                print("Old best val AUC: ", best_auc)
                print("New best val AUC: ", val_auc)
                best_auc = val_auc
                torch.save(model.state_dict(), './'+ model_name +'.pt')
                print("New best model found. Saving now.")
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1
            if epochs_no_improvement == early_stopping_patience:
                print("No decrease in validation loss in %d epochs. Stopping"
                      " training early." % early_stopping_patience)
                break
        if ((max_rate_decreases > 0 and epochs_no_improvement > 0) and
                (epochs_no_improvement % rate_decay_patience == 0) and
                (num_rate_decreases < max_rate_decreases)):

            print("No increase in validation AUC score in %d epochs. "
                  "Reducing learning rate." % rate_decay_patience)

            print("Old learning rate:", learning_rate)
            learning_rate = learning_rate/2.0
            num_rate_decreases += 1

            print("New learning rate:", learning_rate)
            if reload_best_on_decay:
                model.load_state_dict(torch.load('./'+ model_name +'.pt'))
            if optimizer_type == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_type == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
            else:
                raise ValueError("ERROR: optimizer " + str(optimizer_type) +
                                 " not supported.")
            model = model.eval()
            val_acc, \
            val_recall, \
            val_specif, \
            auc, \
            val_loss = utils.check_metrics(model, val_loader, verbose=False)
            model = model.train()

            print("Validation AUC: ", auc)
            print("Validation Loss: ", val_loss)
        print("Epochs without improvement: ", epochs_no_improvement)


        if add_noise in ('input_params',
                         'all_params',
                         'weights_only') and epoch != (epochs-1):

            torch.manual_seed(MANUAL_SEED+epoch)
            model = jitter_params(model=model,
                                  params_to_jitter=add_noise,
                                  scale_term=noise_scale_term)

    print('Finished Training')
    model.load_state_dict(torch.load('./'+ model_name +'.pt'))
    model = model.eval()
    val_acc, \
    val_recall, \
    val_specif, \
    auc, \
    val_loss, \
    val_thresh = utils.check_metrics(model,
                               val_loader,
                               verbose=False,
                               return_threshold=True)

    validation_accs.append(val_acc)
    validation_losses.append(val_loss)
    validation_recalls.append(val_recall)
    validation_specifs.append(val_specif)
    validation_aucs.append(auc)
    model = model.train()

    training_acc, \
    _, \
    _, \
    training_auc, \
    training_loss = utils.check_metrics(model, train_loader, verbose=False)

    training_accs.append(training_acc)
    training_losses.append(training_loss)
    training_aucs.append(training_auc)

    if verbose:
        #print("Training accuracy for epoch: ", train_acc)
        print("validation accuracy: ", val_acc)
        print("validation loss: ", val_loss)
        print("validation recall: ", val_recall)
        print("validation specificity: ", val_specif)
        print("validation AUC: ", auc)
        if return_thresh:
            metrics = (training_accs, validation_accs, training_losses,
                       training_aucs, validation_losses, validation_recalls,
                       validation_specifs, validation_aucs, val_thresh)
        else:
            metrics = (training_accs, validation_accs, training_losses,
                       training_aucs, validation_losses, validation_recalls,
                       validation_specifs, validation_aucs)

    return model, metrics

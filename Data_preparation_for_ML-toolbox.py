# Prepare datasets for ML
from collections import OrderedDict
import numpy as np
import random

# Load datasets from CSV
import sys
import pandas as pd
from PIL import Image

#------------------------------------------------------------#
# Prepare datasets for ML
#------------------------------------------------------------#

def train_test_split(dataset, test_size = 1/2.5, seed = 0):
    ''' 
    Performs a randomized split of a dataset into a training- and test-set for each individual class (key) in the dataset.
    
    Parameters
    ----------
    dataset: dict([<class>, <list of IDs>)])
    test_size: fraction of test-set to training-set (default: 40%)
    seed: starting seed for random.sample function (default: 0)
    
    Returns
    -------
    Tuple of two dictionary containing training and test sets as dict([(<class>, <list of IDs>)]).
    '''
    # set seed of random function
    random.seed(seed)
    
    # create training set and test set
    exp_set_train = OrderedDict()
    exp_set_test = OrderedDict()
    
    for chem, values in dataset.items():
        # determine test set length and sample random experiments
        test_len = int(len(values)*test_size)
        test_ind = random.sample(range(0,len(values)), test_len)
        
        # sort test and training sets by classes
        exp_set_test[chem] = [values[ind] for ind in test_ind]
        exp_set_train[chem] = [values[ind] for ind in list(set(range(0,len(values)))  - set(test_ind))]
        
    return (exp_set_train, exp_set_test)


def sort_by_set(exp_data, *sets):
    '''
    Sort experimental data into training- and test-sets based on datasets obtained by train_test_split.
    
    Parameters
    ----------
    exp_data: dict([(<experiment ID>, <data array>)])
    sets: sequence of datasets as dict([(<class>, <list of IDs>)])
    
    Returns
    -------
    List of lists containing pairs of lists data and labels as [[data_set1],[labels_set1],...]
    '''
    # create empty dictionaries
    sorted_dataset = []
    for dataset in sets:
        data = []
        labels = []

        for chem, set_numbers in dataset.items():
            for num in set_numbers:
                data.append(exp_data[num]/np.sqrt(np.sum(exp_data[num]**2)))
                labels.append(chem)

        sorted_dataset.append(data)
        sorted_dataset.append(labels)
        
    return sorted_dataset


#------------------------------------------------------------#
# Load datasets from CSV
#------------------------------------------------------------#

def load_set_from_CSV(exp_set, folder='../exp_csv'):
    '''
    Loads data from a CSV file containing x-values in column one and y-values in column 2 and beyond.

    CSV format:
        |  x  | y_1 | y_2 | ... |
        |-----|-----|-----|-----|
        |  0  | a_1 | b_1 |     |
        | ... | ... | ... |     |
        |  n  | a_n | b_n |     |
    Files should be labeled in the format: 000006.csv

    Parameters
    ----------
    exp_set: dict([(<class>, <[experiment IDs]>)])
    folder (optional): folder string containing experimental files (default: '../exp_csv')

    Returns
    -------
    Exp_data: dictionary of IDs and exp. data as dict([(<ID>, <np.array(data)>])
    Exp_labels: dictionary of IDs and corresponding class labels as dict([(<ID>, <class>])
    Size of the experimental dataset
    '''
    # prepare dictionaries to reference data
    exp_data = {}
    exp_num_chem = {}

    # start toolbar
    sys.stdout.write("Loading experimental data \n")
    sys.stdout.flush()
    loaded = 0

    for count, chem in enumerate(exp_set):
        nums = exp_set[chem]

        #loads image then normalizes data
        datas = []
        for b,num in enumerate(nums):
            data = pd.read_csv(folder + "/%06d.csv" % (num)) # load data from file
            data_spectra = (data.values[:,1:]-np.min(data.values[:,1:])).transpose()
            datas.append(data_spectra)

            # update the bar
            sys.stdout.write('\r')
            bar = ((b+1)*100)/len(nums)
            # the exact output you're looking for:
            sys.stdout.write(str(chem) + ": [%-20s] %d%%" % ('='*int(bar/5-1) + str(int(bar % 10)), bar))
            sys.stdout.flush()
            loaded += 1
        
        #saves the data in the dictionary
        for n,num in enumerate(nums):
            exp_data[num] = datas[n]
            exp_num_chem[num] = chem
            
        # update the bar
        sys.stdout.write("   " + str(count+1) + "/" + str(len(exp_set)) + " complete\n")
        sys.stdout.flush()

    exp_set_size = loaded
    print("Length of experimental set loaded: " + str(exp_set_size))

    return (exp_data, exp_num_chem, exp_set_size)


def load_images_from_CSV(exp_set, dim = 299, folder='../exp_csv'):
    '''
    Loads 2-dimensional data into a resized image from a CSV file containing x-values in column one and y-values in column 2 and beyond. This is useful when using convolutional neural networks (CNNs).

    CSV format:
        |  x  | y_1 | y_2 | ... |
        |-----|-----|-----|-----|
        |  0  | a_1 | b_1 |     |
        | ... | ... | ... |     |
        |  n  | a_n | b_n |     |
    Files should be labeled in the format: 000006.csv

    Parameters
    ----------
    exp_set: dict([(<class>, <[experiment IDs]>)])
    dim: dimension of the resized image (default: (299,299)
    folder (optional): folder string containing experimental files (default: '../exp_csv')

    Returns
    -------
    Exp_data: np.array of all two-dimensional images with size (dim[0],dim[1],<number of datasets>)
    Exp_labels: list of labels corresponding to the images
    Size of the experimental dataset
    '''
    # prepare dictionaries to reference data
    exp_images = {}
    exp_data = {}
    exp_num_chem = {}

    # prepare arrays to story data
    train_data = []
    train_im = []
    labels = []
    exp_num = []

    # start toolbar
    sys.stdout.write("Loading experimental data \n")
    sys.stdout.flush()
    loaded = 0

    for count, chem in enumerate(exp_set):
        nums = exp_set[chem]

        #loads image then normalizes data
        for b, num in enumerate(nums):
            data = pd.read_csv(folder + "/%06d.csv" % (num)) # load data from file
            data_im = (data.values[:,1:]-np.min(data.values[:,1:])).transpose()
            data_im = data_im*255/np.max(data_im)

            img = Image.fromarray(data_im)
            img = img.convert('RGB')
            img_resized = img.resize(dim) # necessary for Inception module

            exp_images[num] = data_im # keep unconverted data for reference
            exp_num_chem[num] = chem # keep reference of chemical
            exp_data[num] = np.array(img_resized)*255/np.max(img_resized)
            exp_num.append(num)
            train_im.append(img_resized)
            train_data.append(np.array(img_resized)*255/np.max(img_resized))
            labels.append(chem) # create labels

            # update the bar
            sys.stdout.write('\r')
            bar = ((b+1)*100)/len(nums)
            # the exact output you're looking for:
            sys.stdout.write(chem + ": [%-20s] %d%%" % ('='*int(bar/5-1) + str(int(bar % 10)), bar))
            sys.stdout.flush()
            loaded += 1

        # update the bar
        sys.stdout.write("   " + str(count+1) + "/" + str(len(exp_set)) + " complete\n")
        sys.stdout.flush()

    train_data = np.array(train_data)

    exp_set_size = loaded
    print("Length of experimental set loaded: " + str(exp_set_size))
    
    return (train_data, labels, exp_set_size)
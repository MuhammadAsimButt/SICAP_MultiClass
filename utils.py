#./utils.py

import datetime
import pygame
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import openslide
from openslide import deepzoom
import numpy as np

import os
import random
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from torchvision.utils import make_grid
import cv2
from PIL import Image
import pathlib
from pathlib import Path
import pywt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
from collections.abc import Iterable
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import torchmetrics
# Assigning colors to unique values using a colormap
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

matplotlib.use('TkAgg')
#matplotlib.use('Agg')
# This sets the `TkAgg` backend, which is a commonly used backend for displaying plots.

#
plt.interactive(False)


def show(img, closeWindow=False):
    '''
    Displays a single image
    img: tensor
    '''
    npimg = img.numpy()#converts a PyTorch tensor img to a NumPy array
    npimg = npimg.astype(np.uint8)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()
    if(closeWindow):
        # pause for 1 seconds
        plt.pause(1)
        # close the window
        plt.close()


def show_dataset(dataset, n=6):
    '''
    Displays images present in a dataset in the form of a grid(B/n, n). Where B is batch size
    '''
    imgs = [dataset[i][0] for i in range(n)]
    grid = make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt



def show_dl(dl, num_rows=5):
    '''
    Display images present in a batch returned by a dataloader
    num_rows: num of images per screen
    '''

    # Iterate over the dataloaders simultaneously
    for (images, masks) in iter(dl):
        '''
        print(f'image[0] max = {torch.max(images[0])} and min ={torch.min(images[0])}')
        #max = 1.439998984336853 and min =-5.956810474395752
        # Normalize the tensor
        normalized_tensor = (images[0] - torch.min(images[0])) / (torch.max(images[0]) - torch.min(images[0]))
        print("Normalized tensor:", normalized_tensor)
        '''
        # Display images and masks side by side
        num_images = images.shape[0]
        num_screens = (num_images + num_rows - 1) // num_rows
        num_cols = 2

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows*3))
        fig.subplots_adjust(hspace=0.4)

        for i in range(num_screens):
            for j in range(num_rows):
                row_idx = j

                ax_img = axes[row_idx, 0]
                ax_mask = axes[row_idx, 1]

                if (i*num_rows) + j < num_images:
                    index = (i*num_rows)+j
                    normalized_tensor = (images[index] - torch.min(images[index])) / (torch.max(images[index]) - torch.min(images[index]))
                    ax_img.imshow(normalized_tensor.permute(1, 2, 0))
                    ax_img.axis('off')

                    ax_mask.imshow(masks[index].squeeze(), cmap='jet', alpha=0.5)# Use 'jet' colormap
                    ax_mask.axis('off')
                else:
                    ax_img.axis('off')
                    ax_mask.axis('off')

        plt.tight_layout()
        plt.show()

##########################################################################
def split_images_dir(images_dir, train_pct, val_pct, test_pct, imgSubListLength=None):
    """
    Splits a directory of images into three lists of paths containing train, validate, and test images
    according to passed percentages.

    Args:
    - images_dir (str): the directory containing the images
    - train_pct (float): the percentage of images to use for training (between 0 and 1)
    - val_pct (float): the percentage of images to use for validation (between 0 and 1)
    - test_pct (float): the percentage of images to use for testing (between 0 and 1)

    Returns:
    - train_images (list): a list of paths to the images to use for training
    - val_images (list): a list of paths to the images to use for validation
    - test_images (list): a list of paths to the images to use for testing
    """
    #print(f'images dir ={images_dir}')
    #print(f"train_pct ={train_pct},val_pct = {val_pct}, test_pct={type(test_pct)}, sum={train_pct+val_pct+test_pct}")
    assert round((train_pct + val_pct + test_pct),5) == 1.0, "The sum of train_pct, val_pct, and test_pct must be equal to 1.0"

    allFiles_list = os.listdir(images_dir)
    #print(f"[utils]:all files = {len(allFiles_list)}")
    # Do something to filter all images, no file other than images should be included in this list
    images_list = [file for file in allFiles_list if ((file.endswith('.jpg') or file.endswith(".png")))]
    #print(f"[utils]:images_list length = {len(images_list)}")
    images_list = [ os.path.join(images_dir, img) for img in images_list]
    all_images_list = images_list
    #print(f"[utils]:full path images_list length = {len(images_list)}")
    #####################################################
    # Use a portion of the file paths to create a smaller dataset
    if imgSubListLength == None:
        imgSubListLength = len(images_list)

    images_list = images_list[:imgSubListLength]  # Use the first n file paths
    #print(f"[utils]:Adjusted full path images_list length = {len(images_list)}")
    ####################################################
    num_images = len(images_list)
    train_num = int(num_images * train_pct)
    val_num = int(num_images * val_pct)
    test_num = num_images - train_num - val_num

    random.shuffle(images_list)

    train_images = images_list[:train_num]
    val_images = images_list[train_num:train_num+val_num]
    test_images = images_list[train_num+val_num:]

    return train_images, val_images, test_images, all_images_list


#########################################################################
def change_last_directory(paths_list, newDirName):
    """
    Changes the last directory of a list of file paths to "newDirName" and returns the modified list.

    Args:
    - paths_list (list): a list of file paths
    - newDirName: The name of directory to be placed in place of last directory of paths_list

    Returns:
    - modified_paths (list): a list of file paths with the last directory changed to "newDirName"
    """
    #print(f'[utils.py, change_last_directory()]: type of arg paths_list = {type(paths_list)}, and legth ={len(paths_list)}')
    # Convert paths_list to a list if it's a string containing a single path
    if isinstance(paths_list, str):
        paths_list = [paths_list]

    modified_paths = []
    #if list contains paths with file names as last entry e.g xzcz/zxczx/adsf.png

    for path in paths_list:
        #print(f'[utils.py, change_last_directory()]: type of path ={type(path)}, and value={path}') # <class 'str'>
        # Split the path into directory and filename components
        directory, filename = os.path.split(path)
        #print(f'Directory: {directory}, Filename: {filename}')

        # Get the parent directory by splitting the directory path
        parent_directory = os.path.dirname(directory)

        # Create the new path by joining the parent directory and the new directory and file name
        new_path = os.path.join(parent_directory, newDirName, filename)
        #print(f'[utils.py, change_last_directory()]: New Path = {new_path}')

        # Append this new file path to the list of modified paths
        modified_paths.append(new_path)

    #print(f"[change_last_directory]:Last path= {modified_paths[-1]}")

    return modified_paths
##########################################################################
def change_extension(file_paths, new_extension):
    """
    Changes the extension of all files in a list of file paths to a new extension.

    Args:
    - file_paths (list): a list of file paths, or a single string
    - new_extension (str): the new extension to use (e.g., ".png")

    Returns:
    - new_file_paths (list): a list of file paths with the new extension
    """
    #print(f'[utils]: type of arg file_paths = {type(file_paths)}, and length {len(file_paths)}') # <class 'list'>
    new_file_paths = []
    if isinstance(file_paths, list):
        # Iterate over each path in the list
        for file_path in file_paths:
            path_without_ext = os.path.splitext(file_path)[0]  # get the file path without extension
            new_file_path = path_without_ext + new_extension
            new_file_paths.append(new_file_path)
    elif isinstance(file_paths, str):
        path_without_ext = os.path.splitext(file_paths)[0]  # get the file path without extension
        new_file_path = path_without_ext + new_extension
        new_file_paths.append(new_file_path)
    elif isinstance(file_paths, pathlib.WindowsPath):
        file_path = str(file_paths)# convert to a string obj
        path_without_ext = os.path.splitext(file_path)[0]  # get the file path without extension
        new_file_path = path_without_ext + new_extension
        new_file_paths.append(new_file_path)
    else:
        raise TypeError(f"Input must be a list of paths or a single path string. Got {type(file_paths)}")

    return new_file_paths
##########################################################################
def giveFullPathtoAFile(pathUptoFinalDir, fileName):
    print(f'[utils]:pathUptoFinalDir = {pathUptoFinalDir}, fileName = {fileName}')
    if isinstance(pathUptoFinalDir, str):
        print(f'Yes its a STRING object')
        # Convert string to Path object
        path = Path(pathUptoFinalDir)
        fullPathWithFile = path.joinpath(fileName)
        print(f'[utils]:Got STRING obj and fullPathWithFile =  {fullPathWithFile}')
        return fullPathWithFile
    elif isinstance(pathUptoFinalDir,Path) or isinstance(pathUptoFinalDir,pathlib.WindowsPath):
        fullPathWithFile = pathUptoFinalDir.joinpath(fileName)
        print(f'[utils]:Got Path obj and fullPathWithFile =  {fullPathWithFile}')
        return fullPathWithFile
    else:
        raise TypeError("Argument must be a string or a Path object.")
##########################################################################
def save_list_to_file(file_path, data_list):
    """
    Saves a list to a file.

    Args:
    - file_path (str): the path of the file to save
    - data_list (list): the list to save

    Returns:
    - None
    """
    with open(file_path, 'w') as f:
        for item in data_list:
            f.write(str(item) + '\n')

def read_list_from_file(file_path):
    """
    Reads a list from a file.

    Args:
    - file_path (str): the path of the file to read

    Returns:
    - data_list (list): the list read from the file
    """
    with open(file_path, 'r') as f:
        data_list = [line.rstrip() for line in f]
    return data_list

##################################################
def save_ndarray_as_image(ndarray, filename, folder):
    # create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # determine the color map based on the number of channels
    cmap = None
    if len(ndarray.shape) == 2:
        cmap = 'gray'

    # save the ndarray as an image in the folder
    plt.imsave(os.path.join(folder, filename), ndarray, cmap=cmap)

#################################################
def print_unique_values_and_count(path,num_classes=4, class_mapping = {0: 0, 1: 1, 2: 2, 3:3}):
    '''
    This function takes path as a string representing path of image, path as ndarray or path as tensor

    returns unique values and their frequencies and class_weights


    FUTURE: Please try to return a tensor contaiing class weight s of each image in th ebatch
    '''
    #print(f'path = {path}')
    # Load the image using matplotlib
    if isinstance(path, str):
        # input_data is a path string, so read the data from file
        img = mpimg.imread(path)
    elif isinstance(path, np.ndarray):
        # input_data is a NumPy ndarray, so process the data directly
        img = path
    elif isinstance(path, torch.Tensor):
        img = path.numpy()
    else:
        # handle any other input types
        raise TypeError("Unsupported input type")

    #print(f'Shape of img extracted for class weights = {img.shape}')#(256, 256)
    # Get the unique values in the image
    unique_vals = np.unique(img)
    # Find the unique values and their frequencies
    unique_vals, frequencies = np.unique(img, return_counts=True)
    #print(f'Shape of frequencies ={frequencies.shape}')#(3,)

    # Print the number of unique values and the unique values themselves
    '''
    print("Number of unique values in the image:", len(unique_vals))
    print(f"Unique values in the image: {unique_vals}")
    print(f"frequencies of Unique values in the image: {frequencies}")
    '''
    return (unique_vals, frequencies)
######################################################################################3
def getClassWeights(labels, class_mapping = {0: 0, 1: 1, 2: 2, 3:3}):
    '''
    This function takes a tensor of ground truths possibly of shape (B H W)

    returns class_weights of shape(B Classes), per batch and per ground truth (gt)


    '''
    #print(f'\nShape of labels ={labels.shape}')# torch.Size([B, H, W])
    if isinstance(labels, torch.Tensor):
        # Flatten the labels tensor and convert to long data type
        flattened_labels = labels.reshape(-1).long()

        # Calculate the count of each class
        class_counts = torch.bincount(flattened_labels, minlength=4)  # Assuming 4 classes (0, 1, 2, 3)
        #print(f'class_counts = {class_counts}')

        # Calculate the total number of samples
        total_samples = class_counts.sum()
        #print(f'total_samples = {total_samples}')

        # Calculate the inverse of class frequencies as class weights
        #class_weights = 1.0 / (class_counts.float() + 1e-6)  # Add a small epsilon to avoid division by zero
        # proportion of total pixels
        class_weights = (class_counts.float()/total_samples)  # Add 1 to avoid getting zero
        #print(f'class proportions = {class_weights}, max val = {torch.max(class_weights)}')

        # Normalize the class weights
        #class_weights /= class_weights.sum()
        #print(f'Normalized class proportions = {class_weights}')

        ################################################################
        # Reshape the class weights tensor to match the desired shape (B, Classes)
        class_weights_batch = class_weights.reshape(1, -1).repeat(labels.size(0), 1)
        #print(f'Shape of class_weights_batch ={class_weights_batch.shape}')

        # Calculate the average of each class
        class_weights_batch = torch.mean(class_weights_batch, dim=0, keepdim=True)  # Shape: [1, classes]
        #class_weights_batch = class_weights_batch/torch.min(class_weights_batch)  # Shape: [1, classes]

        ########################
        input_tensor =class_weights_batch
        #print(f'input_tensor={class_weights_batch}')
        # Get the indices that would sort the tensor in ascending order along the last dimension
        sorted_indices = torch.argsort(input_tensor, dim=-1)
        #print(f'sorted_indices = {sorted_indices}')
        # Create a new tensor with the same shape as 'input_tensor' and initialize with zeros
        output_tensor = torch.zeros_like(input_tensor)

        # max value of input is in index contained in sorted_indices[0, 3], this index should get min weight

        if (input_tensor[0, sorted_indices[0, 3]] < (0.50)):
            output_tensor[0, sorted_indices[0, 0]] = 3.25  # this index should get max weight
        elif (input_tensor[0, sorted_indices[0, 3]] < (0.85)):
            output_tensor[0, sorted_indices[0, 0]] = 3.5
        else:
            output_tensor[0, sorted_indices[0, 0]] = 4

        if (input_tensor[0, sorted_indices[0, 2]] < (0.35)):
            output_tensor[0, sorted_indices[0, 1]] = 2.25  # this index should get lower than max weight
        elif (input_tensor[0, sorted_indices[0, 2]] < (0.65)):
            output_tensor[0, sorted_indices[0, 1]] = 2.75
        else:
            output_tensor[0, sorted_indices[0, 1]] = 3

        if (input_tensor[0, sorted_indices[0, 1]] < (0.30)):
            output_tensor[0, sorted_indices[0, 2]] = 1.25  # this index should get lower than max weight
        elif (input_tensor[0, sorted_indices[0, 1]] < (0.45)):
            output_tensor[0, sorted_indices[0, 2]] = 1.75
        else:
            output_tensor[0, sorted_indices[0, 2]] = 2
        # min value of input is in index contained in sorted_indices[0, 0], this index should get max weight
        if (input_tensor[0, sorted_indices[0, 0]] < (0.15)):
            output_tensor[0, sorted_indices[0, 3]] = 0.5  # this index should get min weight
        elif (input_tensor[0, sorted_indices[0, 0]] < (0.30)):
            output_tensor[0, sorted_indices[0, 3]] = 0.85
        else:
            output_tensor[0, sorted_indices[0, 3]] = 1

        class_weights_batch =output_tensor
        #print(f'output_tensor={output_tensor}= {class_weights_batch}')
        ########################
        #### Now class weights per label
        # Get the dimensions of the labels tensor
        B, H, W = labels.size()

        class_weights = []

        for b in range(B):
            # Flatten the labels tensor for the current image and convert to long data type
            flattened_labels = labels[b].reshape(-1).long()

            # Calculate the count of each class for the current image
            class_counts = torch.bincount(flattened_labels, minlength=4)  # Assuming 4 classes (0, 1, 2, 3)

            # Calculate the total number of samples for the current image
            total_samples = class_counts.sum()

            # Calculate the inverse of class frequencies as class weights
            weights = 1.0 / (class_counts.float() + 1e-6)  # Add a small epsilon to avoid division by zero

            # Normalize the class weights
            weights /= weights.sum()

            class_weights.append(weights)

        # Stack the class weights for each image into a single tensor
        class_weights_perLabel = torch.stack(class_weights)
    else:
        # handle any other input types
        raise TypeError("Unsupported input type")

    #print(f'class_weights_batch = {class_weights_batch}')
    #print(f'class_weights_perLabel = {class_weights_perLabel}')

    return class_weights_batch, class_weights_perLabel



def print_unique_values_and_histogram(image_paths):
    for path in image_paths:
        # Open image and convert to grayscale
        #img = Image.open(path).convert('L')
        img = Image.open(path)
        # Convert image to numpy array
        img_array = np.array(img)
        # Calculate number of unique pixel values
        unique_vals = np.unique(img_array)
        num_unique_vals = len(unique_vals)
        # Print number of unique values
        #print(f"{path}: {num_unique_vals} unique values={unique_vals}")

        # Plot histogram
        #plt.hist(img_array.flatten(), bins=256, range=[0, 256])
        #plt.title(f"Histogram of {path}")
        #plt.show()
#########################################################################
def changeLabels(path,newPath,value_map=None):
    """
    This function converts labels of classes from 0,3,4,5 to 0,1,2,3
    and puts all masks with replaced lables in a new directory
    """
    # Define a dictionary to map old values to new values
    value_map_SICAPv2 = {0: 0, 3: 1, 4: 2, 5: 3}

    #key code of gerytch: gg3:0, gg4:1, gg5:2, BN:3, ST:4
    value_map_gerytch = {0: 1, 1: 2, 2: 3, 3: 0, 4: 0}

    value_map = value_map_gerytch

    # Set the directory where your images are located
    dir_path = path

    # Loop through each image in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Open the image using PIL
            img = Image.open(os.path.join(dir_path, filename))
            print((f'shape of the OLD mask {img.size}'))

            img_array = np.array(img)

            # Find the unique values in the array
            unique_values = np.unique(img_array)
            # Print the unique values
            print(f"unique_values before changes ={unique_values}")
            print(f'frequency of {unique_values[0]} = {np.count_nonzero(img_array == unique_values[0])}')

            # Convert the pixel values using the value map
            img = img.point(lambda x: value_map.get(x, x))
            print((f'shape of the new mask {img.size}'))

            # Convert the image to a NumPy array
            img_array = np.array(img)
            # Find the unique values in the array
            unique_values = np.unique(img_array)
            # Print the unique values
            print(f"unique_values after changes ={unique_values}")
            print(f'frequency of {unique_values[0]} = {np.count_nonzero(img_array == unique_values[0])}')

            # Save the modified image
            img.save(os.path.join(newPath, filename))

#########################################################################
def open_n_images_and_print_unique_values(directory_path, n):
    """
    Opens the first n images in the given directory path using PIL and prints
    the unique values present in each image.
    Also prints the shape of the image
    """
    for i, file_name in enumerate(os.listdir(directory_path)):
        if i >= n:
            break
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path) and (file_name.endswith('.jpg') or file_name.endswith(".png")):
            with Image.open(file_path) as img:
                unique_values = set(img.getdata())
                print(f'Unique values in {file_name}: {unique_values}')
                print(f'Shape of the image {file_name}: {img.size}')
########################################
def getFileNameFromPath(filepath):
    # extract the filename from the filepath
    filename = os.path.basename(filepath)
    #print(f'Extracted filename = {filename}')
    return filename
#######################################
def getListOfPaths(directory_path, n, extension='.jpg'):
    """
    Returns the first n images in the given directory path using PIL.
    directory_path: can be text or pathlib Path object
    n : no of files to be included in the list
    """
    image_files = []

    if isinstance(directory_path, str):
        # Path is a string
        count = 0
        for i, file_name in enumerate(os.listdir(directory_path)):
            if i >= n:
                break
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path) and file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(file_path)
                count += 1
        files_with_extension = image_files
    elif isinstance(directory_path, Path):
        # Path is a Path object
        # Get a list of all files in the directory with a specific extension
        files_with_extension = list(directory_path.glob('*'.join(extension)))
        files_with_extension = files_with_extension[:n]
    else:
        # Path is neither a string nor a Path object
        raise TypeError('Path must be a string or a Path object')


    return files_with_extension

def display_images(images, rows, cols):
    '''
    Displays images in a grid.
    images: list of image paths
    '''
    fig, axs = plt.subplots(2, 2)
    for i in range(rows):
        for j in range(cols):
            if (i+j) < len(images):
                print(f'images[{i + j}] = {type(images[i + j])}')
                #print(f'images[{i+j}] = {images[i+j]}')
                img = Image.open(images[i+j])
                axs[i, j].imshow(img)
                axs[i, j].set_title('Axis [0, 0]')
                axs[i].axis('off')
            else:
                axs[i].axis('off')
    '''
    When stacking in two directions, the returned axs is a 2D NumPy array.
    If you have to set parameters for each subplot it's handy to iterate over all subplots in a 2D grid 
    using for ax in axs.flat:
    '''
    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.tight_layout()
    plt.show()

#########################################################################
def display_wavelet_coeffs(originalImage, DWT_Coeffs, wavelet= 'db2', max_lev = 3, label_levels = 3, row=2, col=4, figsize=[10, 10]):
    '''
    This function displays a no of combinations of images and their transforms
    originalImage: Single images or its path. right now only opencv format is done
    DWT_Coeffs: list containing coefficient returned by wavedecn function
    max_lev : how many levels of decomposition to draw
    label_levels : how many levels to explicitly label on the plots
    '''
    if isinstance(originalImage, str):
        # If the input is a string, get the file name from path
        imgFileName = getFileNameFromPath(originalImage)
        originalImage = cv2.imread(originalImage)
        #originalImage = Image.open(originalImage) # returned shape is HW, chk
        #print(f'cve  ={originalImage_cv2.shape} and PIL shape {originalImage.size}')

    else:
        imgFileName = 'anyImage'

    #print(f'DWT_Coeffs type = {type(DWT_Coeffs)}')# <class 'list'>
    #print(f'original image shape {originalImage.shape}, dtype {originalImage.dtype}')#(512, 512, 3), dtype uint8

    # Get just shapes of all the details of DWT coefficients
    shapes = pywt.wavedecn_shapes((originalImage.shape[0], originalImage.shape[1]), wavelet=wavelet, level=max_lev, axes=(0,))
    #print(f' shapes of coeffs = {shapes}')

    # determine the size of the largest image
    max_size = shapes[-1]['d'][0]# size of ist level of DWT coefficients i.e cD1 i.e largest quadrant
    #print(f'max size of the ndarreay = {max_size}')

    # calculate the figure size based on the largest image size
    for l in range(max_lev):
        if l==0:
            fig_width = shapes[l][0]/max_size
        else:
            fig_width = fig_width + shapes[l]['d'][0]/max_size

    fig_width = (10 * (fig_width)) + 2
    fig_height = (10 * (max_size / shapes[-1]['d'][1])) + 2
    #print(f'fig_widt = {fig_width}, fig_height = {fig_height}')

    #print(f'fig row = {row}, fig col = {col}')
    Nr = row
    Nc =1+max_lev
    #print(f'Nr = {Nr}, Nc = {Nc}')

    # create a figure with the calculated size and a grid of subplots
    fig, axs = plt.subplots(nrows=Nr, ncols=Nc, figsize=(fig_width, fig_height))
    fig.suptitle('Image and Multilevel DWT Coefficients', fontsize=20)
    #print(f'axs type ={type(axs)}')#<class 'numpy.ndarray'>
    #print(f'axs shape ={axs.shape}')

    # turn off axis labels and ticks for all subplots
    #for ax in axs.flatten():
    #    ax.axis('off')

    for level in range(0, max_lev + 1):
        if level == 0:
            # show the original image before decomposition
            axs[0, 0].set_axis_off()
            #axs[1, 0].imshow(originalImage, cmap=plt.cm.gray)
            axs[1, 0].imshow(originalImage)
            axs[1, 0].set_title('Image', fontsize=10)
            axs[1, 0].set_axis_off()
            continue

        # compute the 2D DWT
        # c = pywt.wavedec2(originalImage, 'db2', mode='periodization', level=level)
        c = pywt.wavedecn(originalImage, wavelet=wavelet, mode='periodization', level=level, axes=range(originalImage.ndim-1))

        # make an array from coeff list
        arr, slices = pywt.coeffs_to_array(c, axes=range(originalImage.ndim-1))
        #print(f'shape of arr = {arr.shape}')#(512, 512, 3)

        # plot subband boundaries of a standard DWT basis
        # Assume a 2D coefficient dictionary, c, from a three-level transform.
        #c = (c[0], c[1]['ad'], c[1]['da'], c[1]['dd'], c[2]['ad'], c[2]['da'], c[2]['dd'], c[3]['ad'], c[3]['da'], c[3]['dd'])
        cAn = arr[slices[0]]  # cA3, see fig at following link smallest approximation (Low pass img) of image
        #print(f'shape of cAn = {cAn.shape}, data type of approximation = {cAn.dtype}')#e.g (256, 256, 3), float64
        # https://pywavelets.readthedocs.io/en/latest/ref/dwt-coefficient-handling.html
        #cD1 = arr[slices[max_lev]['dd']]  # cD1,largest isolation coefficients, 1st HP filter

        #cAn = c[0]# this will change with each level i.e for level 1 it will give largest approximation and so on
        #print(f'data type of approximation = {cAn.dtype}')# as first element of coeff list

        draw_2d_wp_basis((originalImage.shape[0], originalImage.shape[1]),
                         wavedec2_keys(level),
                         ax=axs[0, level],
                         label_levels=label_levels
                         )
        axs[0, level].set_title('{} level\ndecomposition'.format(level))

        # normalize each coefficient array independently for better visibility
        #print(f'max value in cAn before normalization = {np.abs(cAn).max()}')
        #c[0] /= np.abs(c[0]).max()
        cAn /= np.abs(cAn).max()
        #print(f'max value in cAn after normalization = {np.abs(cAn).max()}')
        #print(f'max of ad = {np.abs(arr[slices[0 + 1]["ad"]]).max()}')
        #print(f'max of ad = {np.abs(arr[slices[0 + 1]["da"]]).max()}')
        #print(f'max of ad = {np.abs(arr[slices[0 + 1]["dd"]]).max()}')
        for detail_level in range(level):
            arr[slices[detail_level+1]['ad']] = [d / np.abs(arr[slices[detail_level+1]['ad']]).max() for d in arr[slices[detail_level+1]['ad']]]
            arr[slices[detail_level + 1]['da']] = [d / np.abs(arr[slices[detail_level+1]['da']]).max() for d in arr[slices[detail_level+1]['da']]]
            arr[slices[detail_level + 1]['dd']] = [d / np.abs(arr[slices[detail_level+1]['dd']]).max() for d in arr[slices[detail_level+1]['dd']]]

            arr[slices[detail_level + 1]['ad']] = arr[slices[detail_level + 1]['ad']]*255
            arr[slices[detail_level + 1]['da']] = arr[slices[detail_level + 1]['da']]*255
            arr[slices[detail_level + 1]['dd']] = arr[slices[detail_level + 1]['dd']]*255
            #c[detail_level + 1] = [d / np.abs(d).max() for d in c[detail_level + 1]]
            #print(f'arr[slices[detail_level*3 - 2]["dd"] type = {type(arr[slices[detail_level+1]["dd"]])}')

        #axs[1, level].imshow(arr, cmap=plt.cm.gray)
        axs[1, level].imshow(arr)
        axs[1, level].set_title('Coefficients\n({} level)'.format(level))
        axs[1, level].set_axis_off()
        axs[1, level].set_xticks([])
        axs[1, level].set_yticks([])
        # adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.1, hspace=0.5)

    plt.tight_layout()
    plt.show()

    # save the figure to disk
    print(f'imgFileName whos dwt coeffs plot is to be saved ={imgFileName}')
    ###### pass path to save file in args
    imgFilePath = giveFullPathtoAFile('D:\MAB\PyTorch-Deep-Learning\output', imgFileName)
    print(f'imgFilePath whos dwt coeffs plot is to be saved ={imgFilePath} and its type is {type(imgFilePath)}')
    plotName = change_extension(imgFilePath, '_dwt.png')[0]
    print(f'plotName to be saved ={plotName}')
    plt.savefig(plotName)


    return plt
#########################################################################

def display_images(images, transforms, masks, size=256):
    num_images = len(images)
    fig, axs = plt.subplots(nrows=num_images, ncols=3, figsize=(10, 10))

    for i in range(num_images):
        # Load the original image.
        img = Image.open(images[i])

        # Apply the transform.
        img = transforms[i](img)

        # Load the mask.
        mask = Image.open(masks[i])

        # Resize the image and mask.
        img = img.resize((size, size))
        mask = mask.resize((size, size))

        # Display the original image in the first column.
        axs[i][0].imshow(img)
        axs[i][0].axis('off')
        axs[i][0].set_title('Original')

        # Display the transformed image in the second column.
        axs[i][1].imshow(transforms[i](img))
        axs[i][1].axis('off')
        axs[i][1].set_title('Transformed')

        # Display the mask in the third column.
        axs[i][2].imshow(mask, cmap='gray')
        axs[i][2].axis('off')
        axs[i][2].set_title('Mask')

    plt.tight_layout()
    plt.show()

def displayImgGrid(originalImage, DWT_Coeffs, max_lev = 3, label_levels = 3, row=1, col=4, figsize=[10, 10]):
    print(f'shape of image ={originalImage.shape}')
    im1 = originalImage
    # If the image has an invalid shape, you can transpose the array to fix it
    im1 = np.transpose(im1, (2,0,1))
    print(f'shape of image after transpose ={im1.shape}')
    im2 = im1.T
    im3 = np.flipud(im1)
    im4 = np.fliplr(im2)

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, [im1, im2, im3, im4]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()

#########################################################################
def getImageFrom_ATP(image):
    '''
    image may be an ndarray A, a tensor T, or PIL P

    Returns object that can be displayed by matplotlib
    '''
    if torch.is_tensor(image):
        if (len(image.size()) == 3):
            # transpose the tensor to shape (H, W, C), This can be done in two steps
            # x_transposed = torch.transpose(image, 1, 2)  # swap the 1st and 2nd dimensions CHW to CWH
            # image = torch.transpose(x_transposed, 0, 2)  # swap the 1st and 3rd dimensions CWH to HWC
            image = image.numpy()  # converts a PyTorch tensor img to a NumPy array
            image = image.astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))
        elif (len(image.size()) == 2):
            image = image.numpy()
    elif isinstance(image, np.ndarray):
        # Required shape is (H, W, C)
        if (len(image.shape) == 3):
            # transpose the image data to the correct shape
            if image.shape[2] != 3:
                image = np.transpose(image, (1, 2, 0))
        elif (len(image.shape) == 2):
            print(f'ndarray is a 2D array...')
    elif isinstance(image, Image.Image):
        # img is a PIL image
        pass
    else:
        print(f'Check the type of image and chage code accordingly.')

    return image
#########################################################################
def displayGridOfImagesDynamicaly(images, subplots_per_row = 4, figSize=(12, 8), title=None) -> object:
    '''


    '''
    # Define the number of subplots per row
    subplots_per_row = subplots_per_row

    # Calculate the number of rows and columns needed
    n_plots = len(images)
    n_rows = int(math.ceil(n_plots / subplots_per_row))
    n_cols = subplots_per_row

    # Create the figure and gridspec
    fig = plt.figure(figsize=figSize)
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.4, hspace=0.4)

    # Create and populate the subplots with images
    for i in range(n_plots):
        row = i // subplots_per_row
        col = i % subplots_per_row
        print(f'row = {row}, col = {col}, AND n_row ={n_rows}, n_col={n_cols}')
        ax = fig.add_subplot(gs[row, col])
        img = getImageFrom_ATP(images[i])
        #img = Image.open(f"image{i + 1}.jpg")
        ax.imshow(img)
        ax.set_title(f'Image {i + 1}')

        # Remove ticks from the subplots
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove any empty subplots in the last row
    '''
    if n_plots % subplots_per_row != 0:
        for i in range(subplots_per_row - n_plots % subplots_per_row):
            fig.delaxes(fig.axes[-1])
            print(f'Last Row of Image grid ')
    '''
    # Adjust the spacing between the subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

    return fig
############################################################################
def displayGridOfImagesDynamicaly_3(images,
                                    groundTruth,
                                    aug_img=None,
                                    aug_gt=None,
                                    predLabels=None,
                                    title=None,
                                    subplots_per_row = 3,
                                    row_num_perscreen = 5,
                                    savePath = None,
                                    figSize=(12, 8)) -> object:
    '''


    '''
    #print(f'Shape of input images ={images.shape}')
    # Calculate the number of rows and columns needed
    if len(images) < row_num_perscreen :
        n_rows = len(images)# use of len() is correct
    else:
        n_rows = row_num_perscreen

    n_cols = subplots_per_row
    n_subplots = n_rows*n_cols

    numOfImages = len(images)
    steps = numOfImages // n_rows # only int part is retained and remainder is rejected, need to display full screen
    #print(f'[utils.py]:steps = {numOfImages}//{n_rows} = {steps}')

    # if there are few immages i.e less than images per screen then adjust n-rows
    if steps == 0:
        steps=1

    #print(f'[utils.py]:Screen Number i.e steps = {steps} and Num of images ={numOfImages}')

    ###################
    # Define the color palette
    colors = ['black', 'red', 'green', 'blue']
    cmap = ListedColormap(colors)
    print(f'[displayGridOfImagesDynamicaly_3()]:cmap colors {colors}')
    ###################
    # Define the size of each subplot
    subplot_size = 4

    # Define the minimum and maximum figure sizes
    min_fig_size = 12
    max_fig_size = 18

    # The Figure size on the screen is set by figsize and dpi.
    # figsize is the (width, height) of the Figure in inches
    fig_size = min(max(min_fig_size, n_subplots * subplot_size), max_fig_size)
    #print(f'Figure size= {fig_size}')

    # To make your Figures appear on the screen at the physical size you requested,
    # you should set dpi to the same dpi as your graphics system




    for screenNo in range(steps):
        ########################################
        # Create the figure and gridspec
        fig = plt.figure(figsize=(fig_size, fig_size),
                         facecolor='lightskyblue',
                         layout='constrained'
                         )

        # Other Artists include figure-wide labels (suptitle, supxlabel, supylabel) and text (text).
        fig.suptitle('Images, Ground truth, and Predictions')

        black_patch = mpatches.Patch(color='black', label='Background')
        red_patch = mpatches.Patch(color='red', label='GG3')
        green_patch = mpatches.Patch(color='green', label='GG4')
        blue_patch = mpatches.Patch(color='blue', label='GG5')
        ############################################################
        #fig_size = min(max(min_fig_size, n_subplots * subplot_size), max_fig_size)
        #print(f'fig_size ={fig_size}')
        #print(f'screenNo ={screenNo}')
        clearance = 0.2
        # https://www.geeksforgeeks.org/how-to-create-different-subplot-sizes-in-matplotlib/
        # The GridSpec from the gridspec module is used to adjust the geometry of the Subplot grid.
        # We can use different parameters to adjust the shape, size, and number of columns and rows.
        #

        # to change size of subplot's, set height of each subplot as 4
        fig.set_figheight(8)
        # set width of each subplot as 4
        fig.set_figwidth(8)



        # Calculate the height ratios based on the number of rows
        height_ratios = [1] * n_rows  # Set all height ratios initially to 1
        #print(f'height_ratios = {height_ratios}')
        #total_height = sum(height_ratios)
        #height_ratios = [ratio / total_height for ratio in height_ratios]  # Normalize the ratios
        #print(f'height_ratios after normalization = {height_ratios}')
        if(n_cols==4):
            width_ratios = [1, 1, 1, 1]
        elif(n_cols==3):
            width_ratios = [1, 1, 1]
        elif (n_cols == 2):
            width_ratios = [1, 1]

        # create grid for different subplots
        gs = gridspec.GridSpec(ncols=n_cols,
                               nrows=n_rows,
                               width_ratios=width_ratios,
                               wspace=clearance,
                               hspace=clearance,
                               height_ratios=height_ratios,
                               )

        #gs = fig.add_gridspec(n_rows, n_cols, wspace=clearance, hspace=clearance)

        i=0
        # Create and populate the subplots with images
        for i in range(n_rows):
            row = i
            col = i % subplots_per_row
            #print(f'row = {row}, col = {col}, AND n_row ={n_rows}, n_col={n_cols}')

            # Image
            ax = fig.add_subplot(gs[row, 0],)
            img = getImageFrom_ATP(images[(screenNo*n_rows)+i])
            #img = Image.open(f"image{i + 1}.jpg")
            #ax.imshow(img, cmap=colors)
            ax.set_title(f'Image {i + 1}', loc='left', fontstyle='oblique', fontsize='medium')
            ax.imshow(img)

            # Remove ticks from the subplots
            ax.set_xticks([])
            ax.set_yticks([])

            # groundTruth
            ax = fig.add_subplot(gs[row, 1])
            img = getImageFrom_ATP(groundTruth[(screenNo*n_rows)+i])
            # img = Image.open(f"image{i + 1}.jpg")

            # Display the segmentation mask using the color palette
            ax.imshow(img, cmap=cmap)

            ax.set_title(f'Ground truth {i + 1}', loc='left', fontstyle='oblique', fontsize='medium')
            # Remove ticks from the subplots
            ax.set_xticks([])
            ax.set_yticks([])

            # If augmented image is present
            if (aug_img is not None):
                col = 2
                ax = fig.add_subplot(gs[row, col])
                img = getImageFrom_ATP(aug_img[(screenNo * n_rows) + i])
                # img = Image.open(f"image{i + 1}.jpg")
                ############
                # plt.imshow(arr, cmap=colors)

                ############
                # Display the segmentation mask using the color palette
                ax.imshow(img, cmap=cmap)
                # ax.colorbar(ticks=[0, 1, 2, 3], values=[0, 1, 2, 3])
                # plt.colorbar()
                ax.set_title(f'Augmented Image {i + 1}', loc='left', fontstyle='oblique', fontsize='medium')
                # Remove ticks from the subplots
                ax.set_xticks([])
                ax.set_yticks([])

                # If augmented ground truth is present
                if (aug_gt is not None):
                    col = 3
                    ax = fig.add_subplot(gs[row, col])
                    img = getImageFrom_ATP(aug_gt[(screenNo * n_rows) + i])
                    # img = Image.open(f"image{i + 1}.jpg")
                    ############
                    # plt.imshow(arr, cmap=colors)

                    ############
                    # Display the segmentation mask using the color palette
                    ax.imshow(img, cmap=cmap)
                    # ax.colorbar(ticks=[0, 1, 2, 3], values=[0, 1, 2, 3])
                    # plt.colorbar()
                    ax.set_title(f'Augmented GT {i + 1}', loc='left', fontstyle='oblique', fontsize='medium')
                    # Remove ticks from the subplots
                    ax.set_xticks([])
                    ax.set_yticks([])

            # predLabels, if provided
            if(predLabels is not None):
                col=2
                ax = fig.add_subplot(gs[row, col])
                img = getImageFrom_ATP(predLabels[(screenNo*n_rows)+i])
                # img = Image.open(f"image{i + 1}.jpg")
                ############
                #plt.imshow(arr, cmap=colors)

                ############
                # Display the segmentation mask using the color palette
                ax.imshow(img, cmap=cmap)
                #ax.colorbar(ticks=[0, 1, 2, 3], values=[0, 1, 2, 3])
                #plt.colorbar()
                ax.set_title(f'Prediction {i + 1}', loc='left', fontstyle='oblique', fontsize='medium')
                # Remove ticks from the subplots
                ax.set_xticks([])
                ax.set_yticks([])

        # Remove any empty subplots in the last row
        '''
        if n_plots % subplots_per_row != 0:
            for i in range(subplots_per_row - n_plots % subplots_per_row):
                fig.delaxes(fig.axes[-1])
                print(f'Last Row of Image grid ')
        '''

        # Adjust the spacing between the subplots
        #plt.tight_layout()
        clearance_left = 0.1
        clearance_right = clearance_left*2
        clearance_bottom = 0.1
        clearance_top = clearance_bottom * 2
        plt.subplots_adjust(left=clearance_left,
                            right=clearance_right,
                            bottom=clearance_bottom,
                            top=clearance_top,
                            wspace=clearance,
                            hspace=clearance
                            )
        # Add legend to the figure
        fig.legend(handles=[black_patch, red_patch, green_patch, blue_patch], loc='upper right')
        #plt.legend(loc='upper right')
        # Automatically adjust plot sizes within the figure
        #fig.tight_layout()

        # Displaying the image
        #plt.colorbar()
        plt.show()

        # Save the figure if required
        if savePath is not None:
            # save a PNG formatted figure to the file
            fig.savefig(savePath, dpi=200)

    return fig

############################################################################
def replace_tensor_value_(tensor, a, b):
    tensor[tensor == a] = b
    return tensor


def plot_images(images, groundTruth=None, predLabels=None, title=None, figSize=(8, 8)):
    '''
    images: may be a list of tensors,ndarray or PIL images
    if all image, groundtruth, and predictedLabes are provided they will be displayed side by side
    '''
    num_per_row = 3
    if groundTruth is not None:
        num_rows = len(images)
    else:
        num_rows = int(math.ceil(len(images) / num_per_row))

    print(f'num_rows in the plot = {num_rows}')
    print(f'num of images in Last row = {len(images)%num_per_row}')
    print(f'num of images in the list = {len(images)}')


    # is used to automatically adjust the padding and spacing of the content within a matplotlib figure so that it
    # fits within the boundaries of the figure.
    plt.rcParams.update({'figure.autolayout': True})

    # figure size of 8 inches by 8 inches.
    fig, axes = plt.subplots(num_rows, num_per_row, figsize=figSize, dpi=150)
    #print(f'type of fig={type(fig)}, type and shape of axes ={type(axes)}, {axes.shape}')

    # Define font properties
    #font families = ['Serif', 'Sans-serif', 'Monospace', 'Script']
    font_properties = {'family': 'Serif', 'fontsize': 10, 'fontweight': 'bold'}
    #csfont = {'fontname': 'Arial'}

    # Set the title with custom font properties and wrap the text
    #fig.suptitle(title[0], fontdict=font_properties, wrap=True),
    fig.suptitle(title[0], color='red', wrap=True, **font_properties)#, **csfont)

    # Adjust the spacing to create space below the title
    plt.subplots_adjust(top=3)

    # adjust the spacing between subplots
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    r=0
    c=0
    #for image, ax in zip(images, axes.flat):
    #for image in images:
    for i in range(len(images)):
        row = i // num_per_row
        col = i % num_per_row
        if c==num_per_row:
            c=0 #display ony 3 images per row
            r = r + 1
        image = getImageFrom_ATP(images[r])
        #print(f'type of image ={type(image)}')
        #print(f'shape of image ={image.shape}')

        subPlotFont_properties = {'fontsize': 5, 'fontweight': 'bold', 'fontname': 'Arial'}

        # Display the original image in the first column.
        if groundTruth is not None:
            c=0

        #print(f'Display original image in  r={r}, c={c}')
        #print(f'type of image {type(image)}, shape {image.shape}')#<class 'numpy.ndarray'>, (256, 256, 3)
        axes[r][c].imshow(image)
        axes[r][c].axis('off')
        axes[r][c].set_title(title[1], **subPlotFont_properties)

        # Display ground truth if provided in 2nd col.
        if groundTruth is not None:
            c=2
            gt = getImageFrom_ATP(groundTruth[r])
            print(f'gt type {type(gt)}')
            axes[r][1].imshow(gt)
            axes[r][1].axis('off')
            axes[r][1].set_title(title[2], **subPlotFont_properties)

        # Display the mask in the third column.
        if predLabels is not None:
            c = 2
            pl = getImageFrom_ATP(predLabels[r])
            print(f'pl type {type(pl)}')
            axes[r][2].imshow(pl, cmap='gray')
            axes[r][2].axis('off')
            axes[r][2].set_title(title[3], **subPlotFont_properties)

        c=c+1

    plt.tight_layout()
    # to keep image on the scree
    # display the plot using plt.show()
    plt.show()

    return fig

'''
# Color palette for segmentation masks
PALETTE = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]
    + [[0, 0, 0] for i in range(256 - 22)]
    + [[255, 255, 255]],
    dtype=np.uint8,
)
'''
##########################################################
# MY Color palette for segmentation masks of Gleason Grading
#at index 0 black, 1: red, 2: green, 3: blue, rest of indices = black, last is white
PALETTE = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
    ]
    + [[0, 0, 0] for i in range(256 - 5)]
    + [[255, 255, 255]],
    dtype=np.uint8,
)
###############################################

def array1d_to_pil_image(myarray):
    '''
    takes an ndarray and returns a PIL image
    '''
    # Get the unique values
    # Get the unique values and their frequencies
    unique_values, value_counts = np.unique(myarray, return_counts=True)

    # Print the unique values and their frequencies
    #print(f'[utils.py, array1d_to_pil_image()]: unique_values {unique_values}, frequencies {value_counts}')
    pil_out = Image.fromarray(myarray.astype(np.uint8), mode='P')
    '''
    The mode 'P' stands for "palette mode." In palette mode, each pixel value in the image 
    corresponds to an index in a color palette, where the actual RGB color values are stored
    '''
    pil_out.putpalette(PALETTE)
    return pil_out
########################################################################
def removeBackgroundImages(listOfMasks, imagesDirName, threshold=0.95, classLabel=0, keep=0):
    '''
        Removes images from the list if image only contains background class upto a certain threshhold.
        imagesDirName: Str, Name of directory where images are stored in dataset
        threshold; a percentage, float
        keep: if 0 will drop all candidates, if 1 then will keep 1 after 1 removed , if 2 will 1 after removing 2
        '''
    #print(f'[removeBackgroundImages()]:Type of listOfMasks= {type(listOfMasks)}, and listOfImages ={type(listOfImages)}')
    #<class 'list'> and <class 'list'>
    reducedListOfImages = []
    reducedListOfMasks = []

    kept = 0
    for maskPath in listOfMasks:
        # input_data is a path string, so read the data from file
        #print(f'[removeBackgroundImages()]:maskPath in loop = {maskPath} of type {type(maskPath)}')
        img = mpimg.imread(maskPath)
        #image = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)  # Open the image in grayscale
        #unique_elements = np.unique(img)
        #print(f'unique_elements={unique_elements}')
        #print(f'dtype of image {image.dtype} and dtype of classLabel {type(classLabel)}')
        # Calculate the frequency of the maximum entry (background)
        #print(f'np.min(image)={np.min(image)}, {np.min(image).dtype}')
        #print(f'np.max(image)={np.max(image)}, {np.max(image).dtype}')
        max_entry_freq = np.count_nonzero(img == classLabel)
        #number_of_zeros = np.sum(img == 0.0)
        #print(f'[removeBackgroundImages()]:max_entry_freq = {number_of_zeros}  {max_entry_freq}')

        # Calculate the percentage of the maximum entry frequency
        max_entry_percent = (max_entry_freq / img.size) * 100
        #print(f'[removeBackgroundImages()]:max_entry_percent = {max_entry_percent} and fraction ={max_entry_freq / img.size}')#100.0 and fraction =1.0

        # Check if the percentage less than threshold, lets keep it
        if max_entry_percent <= threshold*100:
            #print(f'\nYes this image can be used for training.')
            reducedListOfMasks.append(maskPath)  # Add the image path to the reduced list

            imgFileName = getFileNameFromPath(maskPath)
            #imgFileName = change_extension(imgFileName, ".jpg")
            imgFilePath = change_last_directory(maskPath,imagesDirName)
            #print(f'Type of imgFilePath ={type(imgFilePath)}, imgFilePath= {imgFilePath}')#<class 'list'>, imgFilePath= ['/home/..4634.png']
            #print(f'Type of imgFilePath[0] ={type(imgFilePath[0])}, imgFilePath= {imgFilePath[0]}')#<class 'str'>,imgFilePath= /home/s...png
            imgFilePathChangedExt = change_extension(imgFilePath, ".jpg")##<class 'list'>, imgFilePathChangedExt= [/home/s...jpg]
            #print(f'Type of imgFilePathChangedExt ={type(imgFilePathChangedExt)}, imgFilePathChangedExt= {imgFilePathChangedExt}')  # <class 'list'>,
            reducedListOfImages.append(imgFilePathChangedExt[0])
        else:
            if kept <= keep:
                kept += 1  # increment by 1
                reducedListOfMasks.append(maskPath)  # Add the image path to the reduced list
                imgFilePath = change_last_directory(maskPath, imagesDirName)
                imgFilePathChangedExt = change_extension(imgFilePath, ".jpg")
                reducedListOfImages.append(imgFilePathChangedExt[0])
            else:
                kept = 0


    #print(f'reducedListOfImages = {reducedListOfImages}')
    #print(f'reducedListOfMasks = {reducedListOfMasks}')

    return reducedListOfImages, reducedListOfMasks
###########################################################
def compare_file_names_twolists(list1, list2):
    file_names1 = [file.split('/')[-1].split('.')[0] for file in list1]
    file_names2 = [file.split('/')[-1].split('.')[0] for file in list2]

    if file_names1 == file_names2:
        print("Both lists have the same file names.")
        print(f'e.g. {list1[1]} = {list2[1]}')
    else:
        print("The file names are different:")
        for index, (name1, name2) in enumerate(zip(file_names1, file_names2)):
            if name1 != name2:
                print(f"Index: {index}, File Name 1: {name1}, File Name 2: {name2}")


#########################
'''
# Assuming you have a tensor called 'input_tensor' with shape (1, 4)
input_tensor = torch.tensor([[5.5, 10, 30, 0]])+1
print(f'input_tensor={input_tensor}')
# Get the indices that would sort the tensor in ascending order along the last dimension
sorted_indices = torch.argsort(input_tensor, dim=-1)
print(f'sorted_indices = {sorted_indices}')
# Create a new tensor with the same shape as 'input_tensor' and initialize with zeros
output_tensor = torch.zeros_like(input_tensor)

# Assign values based on order
'''
'''
output_tensor[0, sorted_indices[0, 0]] = input_tensor[0, sorted_indices[0, 3]]
output_tensor[0, sorted_indices[0, 1]] = input_tensor[0, sorted_indices[0, 2]]
output_tensor[0, sorted_indices[0, 2]] = input_tensor[0, sorted_indices[0, 1]]
output_tensor[0, sorted_indices[0, 3]] = input_tensor[0, sorted_indices[0, 0]]
output_tensor = output_tensor/torch.min(output_tensor)
'''
'''
# max value of input is in index contained in sorted_indices[0, 3], this index should get min weight

if (input_tensor[0, sorted_indices[0, 3]]<((512*512)/4)):
    output_tensor[0, sorted_indices[0, 0]] = 3.25  # this index should get max weight
elif (input_tensor[0, sorted_indices[0, 3]]<((512*512)/2)):
    output_tensor[0, sorted_indices[0, 0]] = 3.5
else:
    output_tensor[0, sorted_indices[0, 0]] = 4

if (input_tensor[0, sorted_indices[0, 2]]<((512*512)/4)):
    output_tensor[0, sorted_indices[0, 1]] = 2.25 # this index should get lower than max weight
elif (input_tensor[0, sorted_indices[0, 2]]<((512*512)/2)):
    output_tensor[0, sorted_indices[0, 1]] = 2.75
else:
    output_tensor[0, sorted_indices[0, 1]] = 3

if (input_tensor[0, sorted_indices[0, 1]]<((512*512)/4)):
    output_tensor[0, sorted_indices[0, 2]] = 1.25 # this index should get lower than max weight
elif (input_tensor[0, sorted_indices[0, 1]]<((512*512)/2)):
    output_tensor[0, sorted_indices[0, 2]] = 1.75
else:
    output_tensor[0, sorted_indices[0, 2]] = 2
# min value of input is in index contained in sorted_indices[0, 0], this index should get max weight
if (input_tensor[0, sorted_indices[0, 0]]<(100/4)):
    output_tensor[0, sorted_indices[0, 3]] = 0.25 # this index should get min weight
elif (input_tensor[0, sorted_indices[0, 0]]<(100/2)):
    output_tensor[0, sorted_indices[0, 3]] = 0.75
else:
    output_tensor[0, sorted_indices[0, 3]] = 1


print(f'output_tensor={output_tensor}')
'''
########################################################################
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 19:05:38 2021

@author: Hp
"""

# If True, display additional NumPy array stats (min, max, mean, is_binary).
ADDITIONAL_NP_STATS = False


def pil_to_np_rgb(pil_img):
    """
    Convert a PIL Image to a NumPy array.
    Note that RGB PIL (w, h) -> NumPy (h, w, 3).
    Args:
      pil_img: The PIL Image.
    Returns:
      The PIL image converted to a NumPy array.
    """
    t = Time()
    rgb = np.asarray(pil_img)
    np_info(rgb, "RGB", t.elapsed())
    return rgb


def np_to_pil(np_img):
    """
    Convert a NumPy array to a PIL Image.
    Args:
      np_img: The image represented as a NumPy array.
    Returns:
       The NumPy array converted to a PIL Image.
    """
    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64":
        np_img = (np_img * 255).astype("uint8")
    return Image.fromarray(np_img)


def np_info(np_arr, name=None, elapsed=None):
    """
    Display information (shape, type, max, min, etc) about a NumPy array.
    Args:
      np_arr: The NumPy array.
      name: The (optional) name of the array.
      elapsed: The (optional) time elapsed to perform a filtering operation.
    """

    if name is None:
        name = "NumPy Array"
    if elapsed is None:
        elapsed = "---"

    if ADDITIONAL_NP_STATS is False:
        print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
    else:
        # np_arr = np.asarray(np_arr)
        max = np_arr.max()
        min = np_arr.min()
        mean = np_arr.mean()
        is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
        print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
            name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))


def display_img(np_img, text=None, font_path="/Library/Fonts/Arial Bold.ttf", size=48, color=(255, 0, 0),
                background=(255, 255, 255), border=(0, 0, 0), bg=False):
    """
    Convert a NumPy array to a PIL image, add text to the image, and display the image.
    Args:
      np_img: Image as a NumPy array.
      text: The text to add to the image.
      font_path: The path to the font to use.
      size: The font size
      color: The font color
      background: The background color
      border: The border color
      bg: If True, add rectangle background behind text
    """
    result = np_to_pil(np_img)
    # if gray, convert to RGB for display
    if result.mode == 'L':
        result = result.convert('RGB')
    draw = ImageDraw.Draw(result)
    if text is not None:
        font = ImageFont.truetype(font_path, size)
        if bg:
            (x, y) = draw.textsize(text, font)
            draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
        draw.text((2, 0), text, color, font=font)
    result.show()


def mask_rgb(rgb, mask):
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.
    Args:
      rgb: RGB image as a NumPy array.
      mask: An image mask to determine which pixels in the original image should be displayed.
    Returns:
      NumPy array representing an RGB image with mask applied.
    """
    t = Time()
    result = rgb * np.dstack([mask, mask, mask])
    np_info(result, "Mask RGB", t.elapsed())
    return result


class Time:
    """
    Class for displaying elapsed time.
    """

    def __init__(self):
        self.start = datetime.datetime.now()

    def elapsed_display(self):
        time_elapsed = self.elapsed()
        print("Time elapsed: " + str(time_elapsed))

    def elapsed(self):
        self.end = datetime.datetime.now()
        time_elapsed = self.end - self.start
        return time_elapsed


##########################################################################
##############################################################################
'''
First we need to write a function to compute the proportion of white pixels 
in the region.

This module will provide right gleason score to each tile for RUMC data provider,
For Karolinska WSI gleason score will be used for each tile becuase no detail is provided in the mask.
'''


def mask_statistics(mask_np, isup_grade, data_provider, gleason_score):
    """
    Args:
        mask   numpy.array   multi-dimensional array of the form WxHxC

    Returns:
        tuple containing
            percent_background   float
            percent_benign
            percent_gleson3
            percent_gleson4
            percent_gleson5
            percent_tissue
            isup_grade
    """
    # print("No of dimensions of ndarray =", image_np.ndim)
    # print(image_np)
    width, height = mask_np.shape[0], mask_np.shape[1]
    num_pixels = width * height
    # print("shape of ndarray", mask_np.shape)
    # print("row col of ndarray", width, height)
    # print("num_pixels = ", num_pixels)
    num_0_pixels = 0
    num_1_pixels = 0
    num_2_pixels = 0
    num_3_pixels = 0
    num_4_pixels = 0
    num_5_pixels = 0

    # Note: A 3-channel white pixel has RGB (255, 255, 255)

    # np.count_nonzero(a): Counts the number of non-zero(more correctly True vales) values
    #                     in the array a

    # Extract red channel only from the mask
    mask_np_red = mask_np[:, :, 0]
    # print("shape of red ch ndarray", mask_np_red.shape)

    num_0_pixels = np.count_nonzero(mask_np_red == 0)  # 0: background (non tissue) or unknown
    # print("num_0_pixels = ", num_0_pixels)
    percent_background = (num_0_pixels / num_pixels) * 100
    # print("percent_background = ", percent_background)

    num_1_pixels = np.count_nonzero(mask_np_red == 1)  # 1: stroma (connective tissue, non-epithelium tissue)
    # print("num_1_pixels = ", num_1_pixels)
    percent_stroma = (num_1_pixels / num_pixels) * 100
    # print("percent_stroma = ", percent_stroma)

    num_2_pixels = np.count_nonzero(mask_np_red == 2)  # 2: healthy (benign) epithelium
    # print("num_2_pixels = ", num_2_pixels)
    percent_benign = (num_2_pixels / num_pixels) * 100
    # print("percent_benign, healthy epithelium = ", percent_benign)

    num_3_pixels = np.count_nonzero(mask_np_red == 3)  # 3: cancerous epithelium (Gleason 3)
    # print("num_3_pixels = ", num_3_pixels)
    percent_gleson3 = (num_3_pixels / num_pixels) * 100
    # print("percent_gleson3 = ", percent_gleson3)

    num_4_pixels = np.count_nonzero(mask_np_red == 4)  # 4: cancerous epithelium (Gleason 4)
    # print("num_4_pixels = ", num_4_pixels)
    percent_gleson4 = (num_4_pixels / num_pixels) * 100
    # print("percent_gleson4 = ", percent_gleson4)

    num_5_pixels = np.count_nonzero(mask_np_red == 5)  # 5: cancerous epithelium (Gleason 5)
    # print("num_5_pixels = ", num_5_pixels)
    percent_gleson5 = (num_5_pixels / num_pixels) * 100
    # print("percent_gleson5 = ", percent_gleson5)

    num_tissue_pixels = np.count_nonzero(np.logical_and(mask_np_red > 0, mask_np_red < 6))
    # print("num_tissue_pixels = ", num_tissue_pixels)
    percent_tissue = (num_tissue_pixels / num_pixels) * 100
    # print("percent_tissue = ", percent_tissue)

    '''
    The grading process consists of finding and classifying cancer tissue 
    into so-called Gleason patterns (3, 4, or 5) based on the architectural 
    growth patterns of the tumor (see Figure below). Based on presence of 
    various formations, the Gleason score is given for majority 
    (first digit in the score) and minority Gleason score (the second digit). 

    After the biopsy is assigned a Gleason score (a combination of the two digits), 
    it is converted into an ISUP grade on a 1-5 scale, using the correspondence 
    matrix shown in the next Figure.
    0+0 or "negative"    0
    3+3(6)      1
    3+4(7)      2
    4+3(7)      3
    4+4(8)      4
    3+5(8)      4
    5+3(8)      4
    4+5(9)      5
    5+4(9)      5
    5+5(10)     5

   	• Radboudumc: Prostate glands are individually labelled. Valid values are:
		○ 0: background (non tissue) or unknown
		○ 1: stroma (connective tissue, non-epithelium tissue)
		○ 2: healthy (benign) epithelium"
		○ 3: cancerous epithelium (Gleason 3)
		○ 4: cancerous epithelium (Gleason 4)
		○ 5: cancerous epithelium (Gleason 5)
	• Karolinska: Regions are labelled. Valid values:
		○ 0: background (non tissue) or unknown
		○ 1: benign tissue (stroma and epithelium combined)
		○ 2: cancerous tissue (stroma and epithelium combined)
    The label masks are stored in an RGB format so that they can be easily opened 
    by image readers. The label information is stored in the red (R) channel, 
    the other channels are set to zero and can be ignored.


    In 2012, the International Society of Urologic Pathologists (ISUP) proposed 
    a novel, validated grading system for clear cell renal cell carcinoma (ccRCC) 
    and papillary renal cell carcinoma (pRCC) that has been implemented by the 
    World Health Organization (WHO).This system is based primarily on the 
    nucleoli assessment of the tumors, as follows [1]:

        Grade 1: Inconspicuous nucleoli at ×400 magnification and basophilic
        Grade 2: Clearly visible nucleoli at ×400 magnification and eosinophilic
        Grade 3: Clearly visible nucleoli at ×100 magnification
        Grade 4: Extreme pleomorphism or rhabdoid and/or sarcomatoid morphology
    '''
    isup_grade_tile = 0
    if (data_provider == 'radboud'):
        percent_cancerous = percent_gleson3 + percent_gleson4 + percent_gleson5
        if (percent_gleson5 > 0 and percent_gleson4 == 0 and percent_gleson3 == 0):
            gleason_score_tile = '5+5'
            isup_grade_tile = 5
        elif (percent_gleson5 > 0 and percent_gleson4 > 0 and percent_gleson4 > percent_gleson5):
            gleason_score_tile = '4+5'
            isup_grade_tile = 5
        elif (percent_gleson5 > 0 and percent_gleson4 > 0 and percent_gleson5 > percent_gleson4):
            gleason_score_tile = '5+4'
            isup_grade_tile = 5
        elif (
                percent_gleson5 > 0 and percent_gleson4 == 0 and percent_gleson3 > 0 and percent_gleson5 > percent_gleson3):
            gleason_score_tile = '5+3'
            isup_grade_tile = 4
        elif (
                percent_gleson5 > 0 and percent_gleson4 == 0 and percent_gleson3 > 0 and percent_gleson3 > percent_gleson5):
            gleason_score_tile = '3+5'
            isup_grade_tile = 4
        elif (percent_gleson5 == 0 and percent_gleson4 > 0 and percent_gleson3 == 0):
            gleason_score_tile = '4+4'
            isup_grade_tile = 4
        elif (
                percent_gleson5 == 0 and percent_gleson4 > 0 and percent_gleson3 > 0 and percent_gleson4 > percent_gleson3):
            gleason_score_tile = '4+3'
            isup_grade_tile = 3
        elif (
                percent_gleson5 == 0 and percent_gleson4 > 0 and percent_gleson3 > 0 and percent_gleson3 > percent_gleson4):
            gleason_score_tile = '3+4'
            isup_grade_tile = 2
        elif (percent_gleson5 == 0 and percent_gleson4 == 0 and percent_gleson3 > 0):
            gleason_score_tile = '3+3'
            isup_grade_tile = 1
        else:
            gleason_score_tile = '0+0'  # 'negative'
            isup_grade_tile = 0
    else:  # Karolinska
        percent_cancerous = percent_benign
        percent_benign = percent_stroma

        if (percent_cancerous > percent_benign):
            gleason_score_tile = gleason_score
            isup_grade_tile = isup_grade
        else:
            gleason_score_tile = '0+0'  # 'negative'
            isup_grade_tile = 0

    # print("tile isup_grade = ", isup_grade)
    mask_mean = mask_np.mean((1, 2))

    red_concentration = mask_mean[0]
    green_concentration = mask_mean[1]
    blue_concentration = mask_mean[2]

    if (data_provider == 'radboud'):
        tile_tuple = (percent_background, percent_stroma, percent_benign,
                      gleason_score, gleason_score_tile,
                      percent_gleson3, percent_gleson4, percent_gleson5,
                      percent_tissue, percent_cancerous, isup_grade, isup_grade_tile)
    else:
        tile_tuple = (percent_background, percent_benign,
                      gleason_score, gleason_score_tile,
                      percent_tissue, percent_cancerous, isup_grade, isup_grade_tile)

    return tile_tuple
###################################################################
def SICAP_mask_statistics(mask_np, which_part=None):
    """
    Args:
        mask   numpy.array   multi-dimensional array of the form WxHxC

    Returns:
        tuple containing
            percent_background   float
            percent_gleson3
            percent_gleson4
            percent_gleson5
            percent_tissue
            isup_grade
    """
    significantDigits=4
    GP_Threshold_patch = 0.00 # 5 percent

    #print(f'No of dimensions of ndarray = {mask_np.ndim} and its shape is = {mask_np.shape}')
    if len(mask_np.shape) == 2:
        # Single channel image
        #print(f'ndarray is 2 dimensional, H = {mask_np.shape[0]}, W= {mask_np.shape[1]}')
        width, height = mask_np.shape[0], mask_np.shape[1]

        num_pixels = width * height # Total pixel of patch, not required

        # Extract red channel only from the mask
        mask_np_red = mask_np
        # print("shape of red ch ndarray", mask_np_red.shape)
        # Get the unique values
        #unique_values = np.unique(mask_np)
        # Print the unique values
        #print(f'\nunique_values in image = {unique_values}')
    elif len(mask_np.shape) == 3 and mask_np.shape[2] == 3:
        # Three channel image
        #print(f'H = {mask_np[0]}, W= {mask_np[1]}, Channels = {mask_np[2]}')
        width, height = mask_np.shape[1], mask_np.shape[2]

        num_pixels = width * height # Total pixel of patch, not required

        # Extract red channel only from the mask
        mask_np_red = mask_np[:, :, 0]
        # print("shape of red ch ndarray", mask_np_red.shape)
    else:
        raise ValueError("Invalid image format. Expected either a single channel or three channel image.")

    if which_part is not None:
        num_pixels_Quarter = (width * height)/4
        if which_part < 1 or which_part > 4:
            raise ValueError("Invalid number of quarters. Expected 1, 2, or 4.")
        else:
            rows, cols = mask_np_red.shape
            if which_part == 1:
                mask_np_red = mask_np_red[:rows // 2, :cols // 2]
                num_pixels = num_pixels_Quarter
                #print(f'Q1 used. Shape= {mask_np_red.shape}')
            elif which_part == 2:
                mask_np_red =  mask_np_red[:rows // 2, :]
                num_pixels = num_pixels_Quarter + num_pixels_Quarter
                #print(f'Q1 and Q2 used. Shape= {mask_np_red.shape}')
            elif which_part == 3:
                mask_np_red =  mask_np_red[:, :cols // 2]
                num_pixels = num_pixels_Quarter + num_pixels_Quarter
                #print(f'Q1 and Q3 used. Shape= {mask_np_red.shape}')
            else:
                mask_np_red = mask_np_red
                num_pixels = width * height
                #print(f'Whole Array used. Shape= {mask_np_red.shape}')

    # print(image_np)

    # print("shape of ndarray", mask_np.shape)
    # print("row col of ndarray", width, height)
    # print("num_pixels = ", num_pixels)
    num_0_pixels = 0
    num_1_pixels = 0
    num_2_pixels = 0
    num_3_pixels = 0

    tile_dict = {
        'percent_background': None,
        'gleason_score_tile': None,
        'primary': None,
        'secondary': None,
        'percent_gleson3': None,
        'percent_gleson4': None,
        'percent_gleson5': None,
        'percent_tissue': None,
        'percent_cancerous': None,
        'isup_grade_tile': None,
        'num_1_pixels' : None,
        'num_2_pixels' : None,
        'num_3_pixels' : None,
        'num_pixels': None,
    }

    num_0_pixels = np.count_nonzero(mask_np_red == 0)  # 0: background (non tissue) or unknown
    #print("num_0_pixels = ", num_0_pixels)
    num_1_pixels = np.count_nonzero(mask_np_red == 1)  # mapped to 3: cancerous epithelium (Gleason 3)
    #print("num_1_pixels = ", num_1_pixels)
    num_2_pixels = np.count_nonzero(mask_np_red == 2)  # 2 mapped to 4: cancerous epithelium (Gleason 4)
    #print("num_2_pixels = ", num_2_pixels)
    num_3_pixels = np.count_nonzero(mask_np_red == 3)  #3 mapped to  5: cancerous epithelium (Gleason 5)
    #print("num_3_pixels = ", num_3_pixels)

    # num_pixels = width * height # Total pixel of patch, not required
    num_cancer_pixels = num_1_pixels + num_2_pixels + num_3_pixels  # Total cancer pixels= GP1+GP2+GP3
    #print("num_cancer_pixels = ", num_cancer_pixels)

    percent_background = round((num_0_pixels / num_pixels) * 100, significantDigits)
    #print("percent_background = ", percent_background)

    if num_cancer_pixels >0:
        percent_gleson3 = round((num_1_pixels / num_cancer_pixels) * 100, significantDigits)
        #print("percent_gleson3 = ", percent_gleson3)

        percent_gleson4 = round((num_2_pixels / num_cancer_pixels) * 100, significantDigits)
        # print("percent_gleson4 = ", percent_gleson4)

        percent_gleson5 = round((num_3_pixels / num_cancer_pixels) * 100, significantDigits)
        # print("percent_gleson5 = ", percent_gleson5)
    else:
        percent_gleson3 = 0
        percent_gleson4 = 0
        percent_gleson5 = 0

    num_tissue_pixels = np.count_nonzero(np.logical_and(mask_np_red > 0, mask_np_red < 4))
    # print("num_tissue_pixels = ", num_tissue_pixels)
    percent_tissue = round((num_tissue_pixels / num_pixels) * 100, significantDigits)
    # print("percent_tissue = ", percent_tissue)


    '''
    0+0 or "negative"    0
    3+3(6)      1
    3+4(7)      2
    4+3(7)      3
    4+4(8)      4
    3+5(8)      4
    5+3(8)      4
    4+5(9)      5
    5+4(9)      5
    5+5(10)     5

   	• SICAPv2: Prostate glands are individually labelled. Valid values are:
		○ 0: background (non tissue) or unknown
		○ 1: cancerous epithelium (Gleason 3)
		○ 2: cancerous epithelium (Gleason 4)
		○ 3: cancerous epithelium (Gleason 5)
    '''
    isup_grade_tile = 0
    primary = 0
    secondary = 0
    gleason_score_tile = ''# to avoid referenced before assignment

    #percent_cancerous = round((percent_gleson3 + percent_gleson4 + percent_gleson5), significantDigits)
    # ALERT: in place of num_pixels num_tissue should be used
    percent_cancerous = round(((num_0_pixels + num_1_pixels + num_2_pixels)/num_pixels), significantDigits)

    if (percent_gleson5 > 0 and percent_gleson4 == 0 and percent_gleson3 == 0):
        if percent_gleson5 > GP_Threshold_patch:
            gleason_score_tile = '5+5'
            isup_grade_tile = 5
            primary = 5
            secondary = 5
        else:
            gleason_score_tile = '0+0'
            isup_grade_tile = 0
            primary = 0
            secondary = 0
    elif (percent_gleson5 > 0 and percent_gleson4 == 0 and percent_gleson3 > 0):
        if(percent_gleson5 > percent_gleson3):
            primary = 5
            if percent_gleson3 > GP_Threshold_patch:
                gleason_score_tile = '5+3'
                isup_grade_tile = 4
                secondary = 3
            else:
                gleason_score_tile = '5+5'
                isup_grade_tile = 5
                secondary = 5
        elif(percent_gleson3 > percent_gleson5):
            primary = 3
            if percent_gleson5 > GP_Threshold_patch:
                gleason_score_tile = '3+5'
                isup_grade_tile = 4
                secondary = 5
            else:
                gleason_score_tile = '3+3'
                isup_grade_tile = 1
                secondary = 3
        else:
            print(f'It seems to be a problem in case 101, G5G4G3')
    elif(percent_gleson5 > 0 and percent_gleson4 > 0 and percent_gleson3 == 0):
        if( percent_gleson4 > percent_gleson5):
            primary = 4
            if percent_gleson5 > GP_Threshold_patch:
                gleason_score_tile = '4+5'
                isup_grade_tile = 5
                secondary = 5
            else:
                gleason_score_tile = '4+4'
                isup_grade_tile = 4
                secondary = 4
        elif (percent_gleson5 > percent_gleson4):
            primary = 5
            if percent_gleson4 > GP_Threshold_patch:
                gleason_score_tile = '5+4'
                isup_grade_tile = 5
                secondary = 4
            else:
                gleason_score_tile = '5+5'
                isup_grade_tile = 5
                secondary = 5
        else:
            print(f'It seems to be a problem in case 110, G5G4G3')
    elif (percent_gleson5 > 0 and percent_gleson4 > 0 and percent_gleson3 > 0):
        if(percent_gleson5 > percent_gleson4 > percent_gleson3):
            primary = 5
            if percent_gleson4 > GP_Threshold_patch:
                gleason_score_tile = '5+4'
                isup_grade_tile = 5
                secondary = 4
            else:
                gleason_score_tile = '5+5'
                isup_grade_tile = 5
                secondary = 5
        elif(percent_gleson5 > percent_gleson3 > percent_gleson4):
            primary = 5
            if percent_gleson3 > GP_Threshold_patch:
                gleason_score_tile = '5+3'
                isup_grade_tile = 4
                secondary = 3
            else:
                gleason_score_tile = '5+5'
                isup_grade_tile = 5
                secondary = 5
        elif (percent_gleson4 > percent_gleson3 > percent_gleson5):
            primary = 4
            if percent_gleson3 > GP_Threshold_patch:
                gleason_score_tile = '4+3'
                isup_grade_tile = 3
                secondary = 3
            else:
                gleason_score_tile = '4+4'
                isup_grade_tile = 4
                secondary = 4
        elif (percent_gleson4 > percent_gleson5 > percent_gleson3):
            primary = 4
            if percent_gleson5 > GP_Threshold_patch:
                gleason_score_tile = '4+5'
                isup_grade_tile = 5
                secondary = 5
            else:
                gleason_score_tile = '4+4'
                isup_grade_tile = 4
                secondary = 4
        elif (percent_gleson3 > percent_gleson5 > percent_gleson4):
            primary = 3
            if percent_gleson5 > GP_Threshold_patch:
                gleason_score_tile = '3+5'
                isup_grade_tile = 4
                secondary = 5
            else:
                gleason_score_tile = '3+3'
                isup_grade_tile = 1
                secondary = 3
        elif (percent_gleson3 > percent_gleson4 > percent_gleson5):
            primary = 3
            if percent_gleson4 > GP_Threshold_patch:
                secondary = 4
                gleason_score_tile = '3+4'
                isup_grade_tile = 2
            else:
                secondary = 3
                gleason_score_tile = '3+3'
                isup_grade_tile = 1
        else:
            print(f'It seems to be a problem in case 111, G5G4G3')
        ###################################################
    elif (percent_gleson5 == 0 and percent_gleson4 > 0 and percent_gleson3 == 0):
        if percent_gleson5 > GP_Threshold_patch:
            gleason_score_tile = '4+4'
            isup_grade_tile = 4
            primary = 4
            secondary = 4
        else:
            gleason_score_tile = '0+0'
            isup_grade_tile = 0
            primary = 0
            secondary = 0
    elif (percent_gleson5 == 0 and percent_gleson4 > 0 and percent_gleson3 > 0):
        if( percent_gleson4 > percent_gleson3):
            primary = 4
            if percent_gleson3 > GP_Threshold_patch:
                secondary = 3
                gleason_score_tile = '4+3'
                isup_grade_tile = 3
            else:
                secondary = 4
                gleason_score_tile = '4+4'
                isup_grade_tile = 4
        elif (percent_gleson3 > percent_gleson4):
            primary = 3
            if percent_gleson4 > GP_Threshold_patch:
                secondary = 4
                gleason_score_tile = '3+4'
                isup_grade_tile = 2
            else:
                secondary = 3
                gleason_score_tile = '3+3'
                isup_grade_tile = 1
        else:
            print(f'It seems to be a problem in case 011, G5G4G3')
            #####################################################
    elif (percent_gleson5 == 0 and percent_gleson4 == 0 and percent_gleson3 > 0):
        if percent_gleson3 >GP_Threshold_patch:
            gleason_score_tile = '3+3'
            isup_grade_tile = 1
            primary = 3
            secondary = 3
        else:
            gleason_score_tile = '0+0'
            isup_grade_tile = 0
            primary = 0
            secondary = 0
        #############################################################
    elif(percent_gleson5 == 0 and percent_gleson4 == 0 and percent_gleson3 == 0):
        gleason_score_tile = '0+0'  # 'negative'
        isup_grade_tile = 0
        primary = 0
        secondary = 0
    else:
        print(f'All combinations are exhausted. Check the problem.')

    # print("tile isup_grade = ", isup_grade)

    tile_dict = {
        'percent_background': percent_background,
        'gleason_score_tile': gleason_score_tile,
        'primary': primary,
        'secondary': secondary,
        'percent_gleson3': percent_gleson3,
        'percent_gleson4': percent_gleson4,
        'percent_gleson5': percent_gleson5,
        'percent_tissue': percent_tissue,
        'percent_cancerous': percent_cancerous,
        'isup_grade_tile': isup_grade_tile,
        'num_1_pixels': num_1_pixels,
        'num_2_pixels': num_2_pixels,
        'num_3_pixels': num_3_pixels,
        'num_pixels':num_pixels,
    }

    return tile_dict


##############################################################################
def slideGrade(GG3_pixels, GG4_pixels, GG5_pixels,
               accum_num_pixels, labeling=0, prediction=0,
               slideNo=None, significantDigits=None
               ):
    if accum_num_pixels >0:
        percent_gleson3=round(((GG3_pixels/accum_num_pixels)*100),significantDigits)
        percent_gleson4=round(((GG4_pixels/accum_num_pixels)*100),significantDigits)
        percent_gleson5=round(((GG5_pixels/accum_num_pixels)*100),significantDigits)
    else:
        percent_gleson3 = 0
        percent_gleson4 = 0
        percent_gleson5 = 0

    if slideNo == '18B0004349G':
        print(f'slideGrade(): percent_gleson3 ={percent_gleson3}')
        print(f'slideGrade(): percent_gleson4 ={percent_gleson4}')
        print(f'slideGrade(): percent_gleson5 ={percent_gleson5}')

    significantDigits = 4

    if labeling==1 and prediction==0:
        GP_Threshold_slide_GP3 = 0.71  #  percent
        GP_Threshold_slide_GP4 = 0.0021  #  percent
        GP_Threshold_slide_GP5 = 0.0150 #0.031  #  percent
    elif labeling==0 and prediction==1:
        GP_Threshold_slide_GP3 =  0.71  #  percent
        GP_Threshold_slide_GP4 =  0.0021  #  percent
        GP_Threshold_slide_GP5 =  0.0150 #0.031  #  percent
    else:
        print(f'[utils.slideGrade()]: There must be only one case either labeling or prediction.')

    percent_cancerous = round((percent_gleson3 + percent_gleson4 + percent_gleson5), significantDigits)

    if (percent_gleson5 > 0 and percent_gleson4 == 0 and percent_gleson3 == 0):
        print(f'100:{percent_gleson5}> {GP_Threshold_slide_GP5}')
        if percent_gleson5 > GP_Threshold_slide_GP5:
            print(f'100:{percent_gleson5}> {GP_Threshold_slide_GP5}')
            gleason_score_slide = '5+5'
            isup_grade_slide = 5
            primary = 5
            secondary = 5
        else:
            gleason_score_slide = '0+0'
            isup_grade_slide = 0
            primary = 0
            secondary = 0
    elif (percent_gleson5 > 0 and percent_gleson4 == 0 and percent_gleson3 > 0):
        if (percent_gleson5 > percent_gleson3):
            primary = 5
            print(f'101:{percent_gleson3}> {GP_Threshold_slide_GP3}')
            if percent_gleson3 > GP_Threshold_slide_GP3:
                print(f'101:{percent_gleson3}> {GP_Threshold_slide_GP3}')
                gleason_score_slide = '5+3'
                isup_grade_slide = 4
                secondary = 3
            else:
                gleason_score_slide = '5+5'
                isup_grade_slide = 5
                secondary = 5
        elif (percent_gleson3 > percent_gleson5):
            primary = 3
            print(f'101:{percent_gleson5}> {GP_Threshold_slide_GP5}')
            if percent_gleson5 > GP_Threshold_slide_GP5:
                print(f'101:{percent_gleson3}> {GP_Threshold_slide_GP5}')
                gleason_score_slide = '3+5'
                isup_grade_slide = 4
                secondary = 5
            else:
                gleason_score_slide = '3+3'
                isup_grade_slide = 1
                secondary = 3
        else:
            print(f'It seems to be a problem in case 101, G5G4G3')
    elif (percent_gleson5 > 0 and percent_gleson4 > 0 and percent_gleson3 == 0):
        if (percent_gleson4 > percent_gleson5):
            primary = 4
            print(f'110:{percent_gleson5}> {GP_Threshold_slide_GP5}')
            if percent_gleson5 > GP_Threshold_slide_GP5:
                print(f'110:{percent_gleson5}> {GP_Threshold_slide_GP5}')
                gleason_score_slide = '4+5'
                isup_grade_slide = 5
                secondary = 5
            else:
                gleason_score_slide = '4+4'
                isup_grade_slide = 4
                secondary = 4
        elif (percent_gleson5 > percent_gleson4):
            primary = 5
            print(f'110:{percent_gleson4}> {GP_Threshold_slide_GP4}')
            if percent_gleson4 > GP_Threshold_slide_GP4:
                print(f'110:{percent_gleson4}> {GP_Threshold_slide_GP4}')
                gleason_score_slide = '5+4'
                isup_grade_slide = 5
                secondary = 4
            else:
                gleason_score_slide = '5+5'
                isup_grade_slide = 5
                secondary = 5
        else:
            print(f'It seems to be a problem in case 110, G5G4G3')
    elif (percent_gleson5 > 0 and percent_gleson4 > 0 and percent_gleson3 > 0):
        if (percent_gleson5 > percent_gleson4 > percent_gleson3):
            primary = 5
            print(f'111:percent_gleson4 {percent_gleson4}> {GP_Threshold_slide_GP4}')
            if percent_gleson4 > GP_Threshold_slide_GP4:
                print(f'111:percent_gleson4 {percent_gleson4}> {GP_Threshold_slide_GP4}')
                gleason_score_slide = '5+4'
                isup_grade_slide = 5
                secondary = 4
            else:
                gleason_score_slide = '5+5'
                isup_grade_slide = 5
                secondary = 5
        elif (percent_gleson5 > percent_gleson3 > percent_gleson4):
            primary = 5
            print(f'111:percent_gleson3 {percent_gleson3}> {GP_Threshold_slide_GP3}')
            if percent_gleson3 > GP_Threshold_slide_GP3:
                print(f'111:percent_gleson3 {percent_gleson3}> {GP_Threshold_slide_GP3}')
                gleason_score_slide = '5+3'
                isup_grade_slide = 4
                secondary = 3
            else:
                gleason_score_slide = '5+5'
                isup_grade_slide = 5
                secondary = 5
        elif (percent_gleson4 > percent_gleson3 > percent_gleson5):
            primary = 4
            print(f'111:percent_gleson3 {percent_gleson3}> {GP_Threshold_slide_GP3}')
            if percent_gleson3 > GP_Threshold_slide_GP3:
                print(f'111:percent_gleson3 {percent_gleson3}> {GP_Threshold_slide_GP3}')
                gleason_score_slide = '4+3'
                isup_grade_slide = 3
                secondary = 3
            else:
                gleason_score_slide = '4+4'
                isup_grade_slide = 4
                secondary = 4
        elif (percent_gleson4 > percent_gleson5 > percent_gleson3):
            primary = 4
            if percent_gleson5 > GP_Threshold_slide_GP5:
                gleason_score_slide = '4+5'
                isup_grade_slide = 5
                secondary = 5
            else:
                gleason_score_slide = '4+4'
                isup_grade_slide = 4
                secondary = 4
        elif (percent_gleson3 > percent_gleson5 > percent_gleson4):
            primary = 3
            if percent_gleson5 > GP_Threshold_slide_GP5:
                gleason_score_slide = '3+5'
                isup_grade_slide = 4
                secondary = 5
            else:
                gleason_score_slide = '3+3'
                isup_grade_slide = 1
                secondary = 3
        elif (percent_gleson3 > percent_gleson4 > percent_gleson5):
            primary = 3
            if percent_gleson4 > GP_Threshold_slide_GP4:
                secondary = 4
                gleason_score_slide = '3+4'
                isup_grade_slide = 2
            else:
                secondary = 3
                gleason_score_slide = '3+3'
                isup_grade_slide = 1
        else:
            print(f'It seems to be a problem in case 111, G5G4G3')
        ###################################################
    elif (percent_gleson5 == 0 and percent_gleson4 > 0 and percent_gleson3 == 0):
        if percent_gleson5 > GP_Threshold_slide_GP5:
            gleason_score_slide = '4+4'
            isup_grade_slide = 4
            primary = 4
            secondary = 4
        else:
            gleason_score_slide = '0+0'
            isup_grade_slide = 0
            primary = 0
            secondary = 0
    elif (percent_gleson5 == 0 and percent_gleson4 > 0 and percent_gleson3 > 0):
        if (percent_gleson4 > percent_gleson3):
            primary = 4
            if percent_gleson3 > GP_Threshold_slide_GP3:
                secondary = 3
                gleason_score_slide = '4+3'
                isup_grade_slide = 3
            else:
                secondary = 4
                gleason_score_slide = '4+4'
                isup_grade_slide = 4
        elif (percent_gleson3 > percent_gleson4):
            primary = 3
            if percent_gleson4 > GP_Threshold_slide_GP4:
                secondary = 4
                gleason_score_slide = '3+4'
                isup_grade_slide = 2
            else:
                secondary = 3
                gleason_score_slide = '3+3'
                isup_grade_slide = 1
        else:
            print(f'It seems to be a problem in case 011, G5G4G3')
            #####################################################
    elif (percent_gleson5 == 0 and percent_gleson4 == 0 and percent_gleson3 > 0):
        if percent_gleson3 > GP_Threshold_slide_GP3:
            gleason_score_slide = '3+3'
            isup_grade_slide = 1
            primary = 3
            secondary = 3
        else:
            gleason_score_slide = '0+0'
            isup_grade_slide = 0
            primary = 0
            secondary = 0
        #############################################################
    elif (percent_gleson5 == 0 and percent_gleson4 == 0 and percent_gleson3 == 0):
        gleason_score_slide = '0+0'  # 'negative'
        isup_grade_slide = 0
        primary = 0
        secondary = 0
    else:
        print(f'All combinations are exhausted. Check the problem.')

    # print("tile isup_grade = ", isup_grade)

    tile_dict = {
        'gleason_score_slide': gleason_score_slide,
        'isup_grade_slide': isup_grade_slide,
        'primary': primary,
        'secondary': secondary,
        'percent_gleson3': percent_gleson3,
        'percent_gleson4': percent_gleson4,
        'percent_gleson5': percent_gleson5,
        'percent_cancerous': percent_cancerous,
    }

    return primary, secondary, isup_grade_slide
def play_beep_sound():
    wave_file = "1kHz_44100Hz_16bit_05sec.wav"
    wave_file_path = os.path.join('/home/hpcladmin/MAB/Projects/PyTorch-Deep-Learning/output/',wave_file)
    pygame.mixer.init()
    beep_sound = pygame.mixer.Sound(wave_file_path)  # Replace "beep.wav" with the path to your beep sound file
    beep_sound.play()


# Define the properties of the tone
duration = 2000  # Duration in milliseconds
frequency = 440  # Frequency in Hz

######################################################################
'''
if __name__ == '__main__':
    
    #path='/home/hpcladmin/MAB/DataSets/SICAPv2/masks'
    #newpath = '/home/hpcladmin/MAB/DataSets/SICAPv2/myMasks'

    path='/home/hpcladmin/MAB/Datasets/C-S DIPC-MA/gleason_grade_20181015/mask_nameless'
    newpath = '/home/hpcladmin/MAB/Datasets/C-S DIPC-MA/gleason_grade_20181015/myMasks'

    anotherPath = '/home/hpcladmin/MAB/DataSets/SICAPv2/masks_Transformed'

    changeLabels(path, newPath=newpath)

    # Print unique values and shape of image
    open_n_images_and_print_unique_values(path, 1)
    open_n_images_and_print_unique_values(newpath, 1)
    #open_n_images_and_print_unique_values(anotherPath, 1)
    
    # Example usage:
    list1 = ['/path/to/image1.jpg', '/path/to/image2.jpg', '/path/to/image3.jpg']
    list2 = ['/path/to/image1.png', '/path/to/image2.png', '/path/to/image3.png']

    compare_file_names_twolists(list1, list2)
    
'''
'''

    data_dir = 'D:\\MAB\\DATASETS\\prostate-cancer-grade-assessment\\train_images\\'
    mask_dir = 'D:\\MAB\\DATASETS\\prostate-cancer-grade-assessment\\train_label_masks\\'

    # image_file = 'ffe06afd66a93258f8fabdef6044e181.tiff'
    # mask_file = 'ffe06afd66a93258f8fabdef6044e181_mask.tiff'
    # data provider = radboud
    # i= 10513
    # gleason score: 0+0 ISUP grade: 0
    # row_tilearray = 12 #12 13 13 13 13 13 13
    # col_tilearray = 94 #95 17 18 19 20 21 24

    # image_file = 'ffaf4de97335595fb711d3985523106a.tiff'
    # mask_file = 'ffaf4de97335595fb711d3985523106a_mask.tiff'
    # i= 10501
    # gleason score: 3+3 ISUP grade: 1
    # row_tilearray = 38 #38 38 40 67 67 67 67
    # col_tilearray = 60 #61 62 59 32 34 35 38

    # image_file = 'ffe236a25d4cbed59438220799920749.tiff'
    # mask_file = 'ffe236a25d4cbed59438220799920749_mask.tiff'
    # i= 10514
    # gleason score: 3+4 ISUP grade: 2
    # row_tilearray = 19 #16 17 17 #18 18 19
    # col_tilearray = 91 #96 95 96 #92 95 91

    # image_file = 'ffac6d538449832cbecba2c4f4908dfa.tiff'
    # mask_file = 'ffac6d538449832cbecba2c4f4908dfa_mask.tiff'
    # i= 10499
    # gleason score: 4+3 ISUP grade: 3
    # row_tilearray = 19 #16 17 17 #18 18 19
    # col_tilearray = 91 #96 95 96 #92 95 91

    # image_file = 'ffe9bcababc858e04840669e788065a1.tiff'
    # mask_file = 'ffe9bcababc858e04840669e788065a1_mask.tiff'
    # i= 10515
    # gleason score: 4+4 ISUP grade: 4
    # row_tilearray = 19 #16 17 17 #18 18 19
    # col_tilearray = 91 #96 95 96 #92 95 91

    image_file = 'ffdc59cd580a1468eac0e6a32dd1ff2d.tiff'
    mask_file = 'ffdc59cd580a1468eac0e6a32dd1ff2d_mask.tiff'
    # i= 10512
    # gleason score: 4+5 ISUP grade: 5
    row_tilearray = 21  # 16 17 17 #18 18 19
    col_tilearray = 6  # 96 95 96 #92 95 91

    image_path = os.path.join(data_dir, image_file)
    mask_path = os.path.join(mask_dir, mask_file)
    print("image_tiff = ", image_path)
    tile_size = 254  # 512

    image = openslide.OpenSlide(image_path)
    mask = openslide.OpenSlide(mask_path)

    zoom_img = openslide.deepzoom.DeepZoomGenerator(image, (tile_size - 2), overlap=1, limit_bounds=False)
    zoom_mask = openslide.deepzoom.DeepZoomGenerator(mask, (tile_size - 2), overlap=1, limit_bounds=False)
    zoom_levels = zoom_img.level_count
    tile_DZ_RGB_PIL_img = zoom_img.get_tile(zoom_levels - 1, (row_tilearray, col_tilearray))
    tile_DZ_RGB_PIL_mask = zoom_mask.get_tile(zoom_levels - 1, (row_tilearray, col_tilearray))

    rgb_img_nd = np.asarray(tile_DZ_RGB_PIL_img)
    rgb_mask_nd = np.asarray(tile_DZ_RGB_PIL_mask)
    print("rgb_mask_nd (only red ch shown) = \n", rgb_mask_nd[:, :, 0])
    # print("rgb_img_nd = \n", rgb_img_nd[:,:,:])
    display_img(rgb_img_nd, text=None, font_path="/Library/Fonts/Arial Bold.ttf", size=48, color=(255, 0, 0),
                background=(255, 255, 255), border=(0, 0, 0), bg=False)
    display_img(rgb_mask_nd, text=None, font_path="/Library/Fonts/Arial Bold.ttf", size=48, color=(255, 0, 0),
                background=(255, 255, 255), border=(0, 0, 0), bg=False)
    mask_statistics(rgb_mask_nd)
'''

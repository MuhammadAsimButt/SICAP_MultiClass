from dataclasses import dataclass
from pathlib import Path
import utils
from PIL import Image
import numpy as np
import os
from scipy import stats
import platform
import torch
import multiprocessing
from poutyne import set_seeds
from poutyne import Model, ModelCheckpoint, CSVLogger

# The @dataclass decorator is a Python feature that is used to automatically generate boilerplate code for
# creating data classes. Data classes provide a concise way to define classes that are primarily used to store data
# and have attributes but do not require explicit methods.
@dataclass
class Project:
    """
    This class represents our project.
    It stores useful information about the structure, e.g. paths.

    base_dir is the path upto dir of current file.
    """

    def __init__(self):
        self.train_images=None
        self.val_images=None
        self.test_images=None
        self.train_masks=None
        self.val_masks=None
        self.test_masks=None
        self.all_images = None
        self.all_masks = None
        self.KFold_train_Val_images = None
        self.KFold_train_Val_masks = None
        self.KFold_test_images = None
        self.KFold_test_masks = None
        self.split = (0.8, 0.1, 0.1)  # train_pct, val_pct, test_pct
        self.KFold_Split = [0.80,0.20]
        self.Keep = 0 # images with greater than a threshold background will be droped
        self.downSampleFactor= None
        ################### For every fold at run tiome  ###########
        self.current_Fold_No = None
        self.train_images_KFold_run = None
        self.train_masks_KFold_run = None
        self.val_images_KFold_run = None
        self.val_masks_KFold_run = None

    #print(torch.version.cuda)
    # Get the number of available CUDA devices
    device_count = torch.cuda.device_count()
    print("Number of CUDA devices:", device_count)

    # Get the properties of each CUDA device
    for i in range(device_count):
        device = torch.device(f"cuda:{i}")
        print("\nProperties of CUDA device", i, ":", device)
        properties = torch.cuda.get_device_properties(device)
        print("  Name:", properties.name)
        print("  Total memory:", properties.total_memory)
        print("  Multiprocessor count:", properties.multi_processor_count)
        print("  Compute capability:", properties.major, ".", properties.minor)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if (device == 'cuda' and device_count>1):
        device = 'all'
    #print(f'device = {device}')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# gives error at some point
    print('The current processor is ...', device)

    # Get the number of available CPU cores
    #num_workers = multiprocessing.cpu_count()
    #num_workers = (multiprocessing.cpu_count())//2 # answer is of type int
    num_workers = 0 # to avoid problems while using DDP
    # Print the number of workers
    print(f'Number of workers:{num_workers}')
    #print(f'Number of workers data type is {type(num_workers)}')
    ##################### FOR DDP #####################################
    n=nodes=1#type=int, metavar='N'
    g=gpus=1# type=int, help='number of gpus per node'
    nr=0#, type=int,      help='ranking within the nodes')
    epochs=2#, type=int,  metavar='N', help='number of total epochs to run')

    world_size = gpus * nodes  #
    os.environ['MASTER_ADDR'] = '10.101.124.11'  #
    os.environ['MASTER_PORT'] = '8888'  #
    #mp.spawn(train, nprocs=args.gpus, args=(args,))  #
    #########################################################

    #################################################################

    Learning_Rate= 0.001
    patience_LR = 3 # reduce the lr on plateau
    patience_ES = 10 # Early stopping
    epochs =800
    batch_size = 8  # to avoid ValueError use batch_size >1
    batch_size_test_dl = 1  #  always keep 1
    num_classes = 4
    img_Mask_size_adjust = 14 # 12 for 6 layers # the size of mask must be cropped to match output size of model so that both can be of same size.
    decoderKernelSize = 3
    SqueezAndExcitation = True
    reduction_ratio = 20

    whichDataLoader = 'SICAPV2_img'  # 'SICAPV2_dwt', 'PANDA_img'

    if whichDataLoader == 'SICAPV2_img':
        img_size = (512, 512)  # when using SICAPV2
        downSampleFactor = None
    elif whichDataLoader == 'SICAPV2_dwt':
        img_size = (512 // 2, 512 // 2)  # when using isolation coeff of dwt
    elif whichDataLoader == 'KFold':
        img_size = (512, 512)

    useDynamicWeights = True #False  # Use True when you need to calculate class weight per batch
    useFixedWeights = False #True  # Use when class weight are to be passed once in the loss function

    NW_used='SemanticSegmentationNet' #'MyCNN' #'PFPN' #
    ##### Regularization ######

    useL1_Regularization = False
    useL2_Regularization = False
    useL2_Regularization_thruOpt = not useL2_Regularization
    useElasticNet_L1plusL2_Regularization = False
    if useL1_Regularization:
        whichRegularization ='L1_Regularization'
    elif useL2_Regularization:
        whichRegularization = 'L2_Regularization'
    elif useL2_Regularization_thruOpt:
        whichRegularization = 'useL2_Regularization_thruOpt'
    elif useElasticNet_L1plusL2_Regularization:
        whichRegularization = 'Elastic_Regularization'
    else:
        whichRegularization = 'No_Regularization'
    # Specify L1 and L2 weights
    l1_weight = 0.2 # start with a small value like 0.001 to prevents the regularization term from dominating the loss function initially
    l2_weight = 0.0000001
    ##### End Regularization ######
    ## Meterics to be monitored by poutyne for training ####
    monitor_metric = "val_loss",#'val_multiclass_cohen_kappa',##'recall'
    monitor_mode = 'min', #'max', # 'max'
    monitor_LR = "val_loss",
    monitor_ES = "val_loss",
    mode_ES = 'min',

    '''
    layerConfig=1
    layer_configs = [
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1},
        {'out_channels': 32, 'kernel_size': 5, 'stride': 1, 'padding': 2, 'dilation': 1},
        {'out_channels': 16, 'kernel_size': 7, 'stride': 1, 'padding': 3, 'dilation': 1},
    ]
    '''
    '''
    layerConfig = 2
    layer_configs = [
        {'out_channels': 64, 'kernel_size': 9, 'stride': 1, 'padding': 1, 'dilation': 1},
        {'out_channels': 32, 'kernel_size': 7, 'stride': 1, 'padding': 3, 'dilation': 1},
        {'out_channels': 16, 'kernel_size': 5, 'stride': 1, 'padding': 3, 'dilation': 1},
        {'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 3, 'dilation': 1},
    ]
    '''
    '''
    layerConfig = 3
    layer_configs = [
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 5, 'kernel_size': 3, 'stride': 1, 'padding': 6, 'dilation': 1},
    ]
    '''
    '''
    layerConfig = 4
    layer_configs = [
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 100, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 24, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
    ]
    '''
    '''
    layerConfig = 5
    layer_configs = [
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 100, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 44, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 30, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 24, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
    ]
    '''
    '''
    layerConfig = 6
    layer_configs = [
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 100, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 44, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 30, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 30, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
    ]
    '''
    '''
    layerConfig = 7
    layer_configs = [
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 100, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 44, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 44, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 44, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
    ]
    '''
    '''
    layerConfig = 8
    layer_configs = [
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
    ]
    '''
    '''
    layerConfig = 7.1
    layer_configs = [
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 110, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 85, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 52, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 48, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
    ]
    '''
    '''
    layerConfig = 7.2
    layer_configs = [
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 100, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 85, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
    ]
    '''
    '''
    layerConfig = 9
    layer_configs = [
        {'out_channels': 130, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 120, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 110, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 100, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 90, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 80, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 70, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
    ]
    '''
    layerConfig = 10
    layer_configs = [
        {'out_channels': 120, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 110, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 100, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 90, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 80, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 70, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
        {'out_channels': 60, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1},
    ]
    layers = len(layer_configs)
    kernelSize = layer_configs[layers-1]['kernel_size']

    useAugmentation = False  # if true then change following list for required transforms
    numOfTransforms = [0, 0, 0, 0, 0, 0, 0]
    useOptions = False


    # FOR DWT
    LEVEL = 1
    WAVELET = 'haar'

    ######## Calculated stats for SICAPv2 #####
    Means_per_channel = [0.82012704, 0.7164683, 0.81525908]
    Standard_deviations_per_channel = [0.14568312, 0.19689716, 0.13686168]
    Skewness_per_channel = [-1.92313286, -0.70942348, -0.97240436]
    Kurtosis_per_channel = [4.82194737, 0.23413237, 0.96273651]

    ################### Class Weights ##################
    # Create a PyTorch tensor from the list with dtype float
    classWeights = torch.tensor([1.0, 1.6, 1.5, 4], dtype=torch.float)# based on prtimary GG in excel file of 156 WSI
    ###############################################
##############################
    # Calculated stats for Imagenet
    #learning_rate = 0.0005
    imagenet_mean = [0.485, 0.456, 0.406]  # mean of the imagenet dataset for normalizing
    imagenet_std = [0.229, 0.224, 0.225]  # std of the imagenet dataset for normalizing
    set_seeds(42)

##############################
    '''
    pathlib module offers classes representing filesystem paths with semantics appropriate for 
    different operating systems. Path classes are divided between 
    pure paths, which provide purely computational operations without I/O, and 
    concrete paths, which inherit from pure paths but also provide I/O operations.
    pathlib's Path instantiates a concrete path for the platform the code is running on.
    '''
    # get directory path of project code
    base_dir: Path = Path(__file__).parents[0]
    #print(f"[project.py]: path object = {Path(__file__)}")
    #data_dir = base_dir / 'dataset'
    #print(f"[project.py]: base_dir = {base_dir}")

   # get directory path of dataset
    if platform.system() == 'Windows':
        # Modify the path for Windows
        data_dir = Path('D:\MAB\DataSets\SICAPv2')  # create a path object
    else:
        # Modify the path for Linux
        #data_dir = Path('/home/senadmin/MAB/Datasets/SICAPv2/')  # create a path object
        data_dir = Path('/home/hpcladmin/MAB/Datasets/SICAPv2/')  # create a path object
    #print(f"[project.py]: data_dir = {data_dir}")

    # Creat paths relative to code directory
    checkpoint_dir = base_dir / 'checkpoint'
    output_dir = base_dir / 'output'
    img_paths_file = output_dir.joinpath('test_img.txt')
    mask_paths_file =output_dir.joinpath('test_mask.txt')
    # if test dataloader is to be save then use following path
    test_dl_pickle = output_dir.joinpath('test_dl.pkl')
    poutyne_Logs = base_dir.joinpath('poutyne_Logs')

    # Creat images dir path relative to dataset directory
    imagesDirName = 'images'
    images_dir = data_dir.joinpath(imagesDirName)

    maskDirName = 'myMasks'#'masks_Transformed'# 'myMasks'
    # Creat paths relative to dataset directory
    mask_dir = data_dir.joinpath(maskDirName)
    # mask_dir = Path('D:\MAB\DataSets\SICAPv2\myMasks')  # create a path object
    #print(f"[project.py]: mask_dir = {mask_dir}")
    transforms_dir = data_dir.joinpath('ImageTransforms')
    '''
    DWT_1_dir = transforms_dir.joinpath('DWT_1')  # create a path object for DWt coef and decomposition level 1
    IDWT_1_dir = transforms_dir.joinpath('IDWT_1')  # create a path object for DWt coef and decomposition level 1
    DWT_2_dir = transforms_dir.joinpath('DWT_2')  # create a path object for DWt coef and decomposition level 2
    IDWT_2_dir = transforms_dir.joinpath('IDWT_2')  # create a path object for DWt coef and decomposition level 2
    FFT_dir = transforms_dir.joinpath('FFT')  # create a path object for FFT coef
    IFFT_dir = transforms_dir.joinpath('IFFT')  # create a path object for FFT coef
    '''
    WSI_Slide_patches_DS_FileName = 'wsi_labels.xlsx'
    WSI_Slide_pathes_DS_path = data_dir.joinpath(WSI_Slide_patches_DS_FileName)
    images_slide_Dir_Name = 'images_slide'
    images_slide_Dir_Path = data_dir.joinpath(images_slide_Dir_Name)
    mask_slide_Dir_Name = 'masks_slide'
    mask_slide_Dir_Path = data_dir.joinpath(mask_slide_Dir_Name)

    dirInPoutyne_Logs = ('NW_' +NW_used+ '_DL_'+whichDataLoader + '_NWLs_' +str(layers)+ '_KS_' +str(kernelSize)+
                         '_BS_' + str(batch_size) + '_W_' + str(useDynamicWeights) +
                         '_A_' + str(useAugmentation)+
                         '_WDyn_' +str(useFixedWeights)+'_LC_' + str(layerConfig)+
                         '_monitor_'+str(monitor_metric[0])+'_NWDecKer_'+str(decoderKernelSize)+
                         '_SE_' + str(SqueezAndExcitation)+'_RR_'+str(reduction_ratio)+
                         '_Reg_'+whichRegularization+'_l2_weight_'+str(l2_weight)
                         )
    def __post_init__(self):
        print(f'\n Oh project object has been created...\n')
        # create the directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        '''
        self.DWT_1_dir.mkdir(exist_ok=True)
        self.IDWT_1_dir.mkdir(exist_ok=True)
        self.DWT_2_dir.mkdir(exist_ok=True)
        self.IDWT_2_dir.mkdir(exist_ok=True)
        self.FFT_dir.mkdir(exist_ok=True)
        self.IFFT_dir.mkdir(exist_ok=True)
        '''
    def preprocessDataset(self,
                          images_dir,
                          maskDirName,
                          imagesDirName,
                          img_paths_file,
                          mask_paths_file,
                          imgSubListLength = None
                          ):
        split = (0.7, 0.2, 0.1)# train_pct, val_pct, test_pct
        self.train_images, self.val_images, self.test_images = utils.split_images_dir(images_dir,
                                                                       split[0],
                                                                       split[1],
                                                                       split[2],
                                                                       imgSubListLength=imgSubListLength
                                                                       )

        # Prepare path lists for corresponding masks for images in above image lists
        #print(f"[project]:train_images list length before removing images with mostly background= {len(self.train_images)}")# Its a List
        #print(f'type of elements of train_images = {type(train_images[0])}')# Str
        self.train_masks = utils.change_extension(utils.change_last_directory(self.train_images,maskDirName), ".png")
        #print(f"[project]:train_masks list length = {len(train_masks)}")  # Its a List
        self.val_masks = utils.change_extension(utils.change_last_directory(self.val_images, maskDirName), ".png")
        self.test_masks = utils.change_extension(utils.change_last_directory(self.test_images, maskDirName), ".png")

        # remove images from training list to improve class imbalance problem
        self.train_images, self.train_masks = utils.removeBackgroundImages(self.train_masks,
                                                                           self.train_images,
                                                                           imagesDirName,
                                                                           threshold=0.95,
                                                                           classLabel=0,
                                                                           keep=1
                                                                           )

        #print(f"[project]:train_images list length after removing images with mostly background= {len(self.train_images)}")
        #print(f"[project]:test_masks list length = {len(self.test_masks)}")  # Its a List
        # Save list of paths for evaluation of model afterwards in other module
        #print(f'Test images path = {img_paths_file}')
        #print(f'Test masks path = {mask_paths_file}')
        utils.save_list_to_file(img_paths_file, self.test_images)
        utils.save_list_to_file(mask_paths_file, self.test_masks)
    ####################################################


####################################################
def calculate_stats(img_dir):
    """
    Calculates the mean, standard deviation, and variance of a PyTorch dataset.

    Args:

    Returns:
        tuple: A tuple containing the mean, standard deviation, and variance.
    """

    # Initialize arrays to store the statistics
    # The array has a shape of (4,), which means it is a 1-dimensional array with 4 elements.
    means = np.zeros((4,))
    stds = np.zeros((4,))
    skews = np.zeros((4,))
    kurts = np.zeros((4,))

    # Loop through each image in the directory
    for filename in os.listdir(img_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Open the image using PIL and convert it to a NumPy array
            img = np.array(Image.open(os.path.join(img_dir, filename)))
            # Divide every pixel by 255 to get values between 0 and 1
            img = img / 255
            #print(f'max value in the array = {max(img)}')
            #print(f'data type of image = {img.dtype}')
            # Calculate the statistics per channel
            for i in range(3):
                channel = img[:, :, i]
                means[i] += np.mean(channel)
                stds[i] += np.std(channel)
                skews[i] += stats.skew(channel.flatten())
                kurts[i] += stats.kurtosis(channel.flatten())

            # Calculate the statistics for the whole image
            means[3] += np.mean(img)
            stds[3] += np.std(img)
            skews[3] += stats.skew(img.flatten())
            kurts[3] += stats.kurtosis(img.flatten())

    # Calculate the average statistics per channel and for the whole directory
    n_images = len(os.listdir(img_dir))
    means /= n_images
    stds /= n_images
    skews /= n_images
    kurts /= n_images

    # Print the results
    #print('Means per channel: ', means[:3])
    #print('Standard deviations per channel: ', stds[:3])
    #print('Skewness per channel: ', skews[:3])
    #print('Kurtosis per channel: ', kurts[:3])

    #print('Mean of all channels: ', means[3])
    #print('Standard deviation of all channels: ', stds[3])
    #print('Skewness of all channels: ', skews[3])
    #print('Kurtosis of all channels: ', kurts[3])

    return means, stds, skews, kurts
# This single object should be imported in every modle
project = Project()
project.preprocessDataset(project.images_dir,
                          project.maskDirName,
                          project.imagesDirName,
                          project.img_paths_file,
                          project.mask_paths_file,
                          #imgSubListLength = 50
                          )
########################################################
# Run test code OR code to br run one time
'''
if __name__ == '__main__':
    project = Project()

    mean, std, skewness, kurtosis = calculate_stats(project.images_dir)
    print(f"dataset mean = {mean[0]}, {mean[1]}, {mean[2]}")
    print(f"dataset std = {std[0]}, {std[1]}, {std[2]}")
    print(f"dataset skewness = {skewness[0]}, {skewness[1]}, {skewness[2]}")
    print(f"dataset kurtosis = {kurtosis[0]}, {kurtosis[1]}, {kurtosis[2]}")
'''
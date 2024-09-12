'''

THIS FILE IS NOW DIRTY SO BEFORE USING IT HAVE A COMPLETE LOOK?????????????
I am goig to do inferece file splits according to slide id in another module , done here too but not tested yet.
Now I am going to use mainly pandas
??????????????
Read text file that contains all patches related to a single slide.
For lists of patches in a sigle row. i.e common y_ini.

This module stores a csv file containing all information about slide classification in output folder of
relevant poutyne log projects folder.

'''
import pandas as pd
import csv
import os
import random
import skimage.io as io
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import cohen_kappa_score
import re
import pickle
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from collections import OrderedDict

#from Project_HomeLapTop import project  # All parameters, paths are stored here
#from Project_CPU import project  # All parameters, paths are stored here
from Project_DDP import project  # All parameters, paths are stored here
import utils
from WSI_Classification_BOW_KNN import KNNClassifier
from SVMClassifier import SVMClassifier
from baysianClassifier import BayesianClassifier

significantDigits= 3

#class WSI_DS(Dataset):
class WSI_DS():
    '''
    This class contain data about all files regarding the dataset
    '''

    def __init__(self,
                 pickleListOfLists_file_name,# input
                 WSI_labels,  # BM file, Slide level labels for help and accuracy calculations, INPUT file
                 featureType='ISUP', #
                 useKNNClassifier = 0,# Use KNNClassifier or not, 2 means use KNNClassifier with your algo
                 augmentDataset = 0,# augment Ds or not at  runtime
                 #reducedBow = 0, # 0: include NC in BOW
                 newSegmentationInference=False,
                 newClassificationInference=False,
                 inference_dir_path = None,  # Segmentation Inferred masks (Source), OR output
                 infer_Classi_patch_file_name=None,# source file in case of inference by classification model ELSE output file.
                 infer_Classi_slide_file_name = None,#output, SLide Level inference( with the help of patch level inference)
                 KNNClassifier_obj=None,
                 KNNClassifier_param_dict=None,
                 ):
        '''
        pickleListOfLists_file_name: A list of lists. Every sublist contains full paths of all img/mask related to a
            single slide. These masks are original dataset masks.
            e.g. /home/hpcladmin/MAB/Datasets/SICAPv2/myMasks/18B000646H_Block_Region_20_1_4_xini_53027_yini_53142.png
        inference_dir_path: If segmentation was used then this directory contains infered masks
        newSegmentationInference:if newSegmentationInference is true then we will extract info from directory containing new inference made by
            a proposed network otherwise we will extract slide classification info from original file and origional
            masks for original dataset correction and future reference.

        newClassificationInference: If new inferences are made by  a classifier and inferences are present in a csv file.
                we will use that csv file of inferences to extract info about every WSI to assign and a new csv file
                will be made for only slide level ISUP grades.

        WSI_labels: Path to Bench mark csv file

        infer_Classi_patch_file_name: Full path of the inference csv file if classification was used for patch inference
        infer_Classi_slide_file_name: This file will contain final ISUP grades after calculating it thru info contained in patches. In case of classification this info is in above file. = None,
        '''
        self.featureType = featureType
        self.useKNNClassifier= useKNNClassifier# 0: dont use, 1: use, 2 use with your own algo
        self.augmentDataset=augmentDataset
        #self.reducedBow = reducedBow
        self.inference_dir_path = inference_dir_path
        self.infer_Classi_patch_file_name=infer_Classi_patch_file_name# source: in case of classification,This file contain inferences of all patches
        self.infer_Classi_slide_file_name = infer_Classi_slide_file_name# destination: in case of classification, slide classification
        self.newSegmentationInference = newSegmentationInference
        self.newClassificationInference= newClassificationInference
        self.WSI_labels =WSI_labels# Slide level labels for help and accuracy calculations, INPUT file path
        self.pickleListOfLists_file_name= pickleListOfLists_file_name
        self.BagOfWords_ISUP = OrderedDict()
        self.BagOfWords_GS = OrderedDict()
        self.BagOfWords_GS_Label = OrderedDict()
        self.BagOfWords_ISUP_Label = OrderedDict()
        self.BagOfWords_df=pd.DataFrame(columns=['Slide_No.',
                                                 'slide_BOW_ISUP_NL',
                                                 'ISUP0', 'ISUP1', 'ISUP2', 'ISUP3', 'ISUP4', 'ISUP5',
                                                 'slide_BOW_GS_NL',
                                                 'GS0','GS1','GS2','GS3','GS4','GS5','GS6','GS7','GS8','GS9',
                                                 'slide_GS_NL',
                                                 'slide_isup_NL',
                                                 'slide_GS_BM',
                                                 'slide_isup_BM']
                                             )
        # Load the saved lists. Full paths to original dataset masks
        self.listoffiles = self.loadList_pickle(self.pickleListOfLists_file_name)
        self.wronglyLabeledSlide_df = pd.DataFrame(columns=['Slide_No.',
                                                            'percent_gleson3','percent_gleson4','percent_gleson5',
                                                            'slide_GS_NL', 'slide_isup_NL',
                                                            'slide_GS_BM', 'slide_isup_BM',
                                                            'slide_BOW_ISUP_NL',
                                                            ]
                                                   )
        self.wronglyPredictedSlide_df = pd.DataFrame(columns=['Slide_No.','BagOfPatches_predicted',
                                                              'percent_gleson3', 'percent_gleson4', 'percent_gleson5',
                                                              #'slide_GS_NL', 'slide_ISUP_NL',
                                                              'slide_GS_BM', 'slide_ISUP_BM',
                                                              'slide_GS_NL', 'slide_ISUP_NL',
                                                              'slide_ISUP_Predicted',]
                                                     )
        # extract sublists from listoffiles and make data framems, then append those dfs in a list
        self.df_list = []
        for i, sublist in enumerate(self.listoffiles):
            #print(f'length of img sublist = {len(sublist)}')
            #print(f'sublist first element  = {sublist[0]}')
            # /home/hpcladmin/MAB/Datasets/SICAPv2/myMasks/18B000646H_Block_Region_20_1_4_xini_53027_yini_53142.png

            # Change full path obtained to full path of new inferences mask folder made by new DNN architecture
            new_sublist = []
            if newSegmentationInference:
                # change paths because we need to access new inference masks for calculations
                for filepath in enumerate(sublist):
                    #print(f'filepath = {filepath}')# tuple
                    file_name = utils.getFileNameFromPath(filepath[1])
                    full_path = os.path.join(self.inference_dir_path,file_name)
                    #print(f' new full path {full_path}')
                    new_sublist.append(full_path)
                sublist = new_sublist
            elif newClassificationInference:
                # no need to chage paths in sublists, we have a csv that contains all
                # inferences about patches
                sublist = sublist
            else:# This case belong to Labeling, So no need to change paths.
                sublist = sublist

            self.df_list.append(self.extract_dataFrame(sublist))

        if self.newSegmentationInference:
            # Both files are output files
            self.csv_filename_patch = 'patch_slideClassification'+'_' +project.NW_used#+'.csv'
            self.csv_filename_slide = 'slideClassification'+'_' + project.NW_used#+'.csv'
            self.WSI_labels_df = pd.read_csv(WSI_labels)  # WSI_labels = BM_File_path, Slide level labels, INPUT file
        elif self.newClassificationInference:
            # source file
            # These two files are now passed in arg and assigned here can be removed in future
            self.csv_filename_patch = self.infer_Classi_patch_file_name
            # destination file, output file that will contain WSI level ISUP grades
            self.csv_filename_slide = self.infer_Classi_slide_file_name
            self.WSI_labels_df = pd.read_csv(WSI_labels)  # WSI_labels = BM_File_path, Slide level labels, INPUT file


            self.KNNClassifier_obj = KNNClassifier(featureType=KNNClassifier_param_dict['featureType'],#'ISUP',
                                                   n_neighbors=KNNClassifier_param_dict['n_neighbors'],
                                                   L_normalization=KNNClassifier_param_dict['L_normalization'],
                                                   weights=KNNClassifier_param_dict['weights'], # 'uniform', 'distance'
                                                   algorithm=KNNClassifier_param_dict['algorithm'],#'auto'
                                                   metric=KNNClassifier_param_dict['metric'],# 'minkowski', 'cityblock', 'cosine',
                                                   p=KNNClassifier_param_dict['p'],
                                                   updateDS=int(self.augmentDataset),
                                                   reducedBow=KNNClassifier_param_dict['reducedBow'],# 1 means do not include NC
                                                   )# To be used for WSI classification based on BOW of patch inference

            #self.KNNClassifier_obj = SVMClassifier(num_classes=6) # BayesianClassifier(num_classes=6)
        else:
            # Here we will use masks of dataset to calculate patch level and slide level ISUPs Labeling based on calculation
            # Both files are output files
            self.csv_filename_patch = self.infer_Classi_patch_file_name #'patch_slideClassification'# output
            # destination file, output file that will contain WSI level ISUP grades
            self.csv_filename_slide = self.infer_Classi_slide_file_name #'slideClassification'# output
            self.WSI_labels_df = pd.read_excel(WSI_labels)  # WSI_labels = BM_File_path, Slide level labels, INPUT file

        if self.newClassificationInference:
            self.headers_slide = ['S.No.', 'Slide_No.','BagOfPatches_predicted',
                                  'percent_gleson3_NL', 'percent_gleson4_NL', 'percent_gleson5_NL',
                                  'Slide_primary_BM', 'Slide_Secondary_BM', 'Slide_ISUP_BM',
                                  'Slide_primary_prop_NL', 'Slide_Secondary_prop_NL', 'Slide_ISUP_prop_NL',
                                  'Slide_ISUP_PredByClassifier',
                                  'BM_PredByClassifier_ISUP_agree', 'propNL_PredByClassifier_agree',
                                  ]
        elif self.newSegmentationInference:
            # Here proposed scheme is that of extracting overlaps in patches. But all extracted data belong to same DNN
            # Chk that data is same in both case with or without overlaps e.g observe percent_gleson3_p and percent_gleson3
            self.headers_patch = ['S.No.', 'Slide_No.', 'Patch_No.',
                                  'percent_gleson3', 'percent_gleson4', 'percent_gleson5',
                                  'percent_gleson3_p', 'percent_gleson4_p', 'percent_gleson5_p',
                                  'Patch_primary', 'patch_Secondary', 'patch_ISUP',
                                  'Slide_primary', 'Slide_Secondary', 'Slide_ISUP',
                                  'Slide_p_prop', 'Slide_S_prop', 'Slide_ISUP_prop'
                                  ]
            self.headers_slide = ['S.No.', 'Slide_No.',
                                  'percent_gleson3', 'percent_gleson4', 'percent_gleson5',
                                  'percent_gleson3_p', 'percent_gleson4_p', 'percent_gleson5_p',
                                  'Slide_primary', 'Slide_Secondary', 'Slide_ISUP',
                                  'Slide_p_prop', 'Slide_S_prop', 'Slide_ISUP_prop',
                                  ]
        else:
            # In case of labeling original dataset Masks will be used to extract info and make patch and WSI csv.
            self.headers_patch = ['S.No.', 'Slide_No.', 'Patch_No.',
                                  'percent_gleson3', 'percent_gleson4', 'percent_gleson5',
                                  'Patch_primary', 'patch_Secondary', 'patch_ISUP',
                                  ]
            self.headers_slide = ['S.No.', 'Slide_No.','BagOfLabels',
                                  'percent_gleson3', 'percent_gleson4', 'percent_gleson5',
                                  'Slide_primary_BM', 'Slide_Secondary_BM', 'Slide_ISUP_BM',
                                  'Slide_primary_prop', 'Slide_Secondary_prop', 'Slide_ISUP_prop',
                                  'Primary_agree', 'Secondary_agree', 'slide_agree',
                                  ]
        self.S_No = 0  # of csv file
        self.slide_No = '00000'  # Slide Number in dataset(list of lists)

        # Classify all slides
        #self.classify_slides(WSI_labels_df=self.WSI_labels_df)

        # Ground truth labels and predicted classes
        self.ground_truth = []
        self.gt_BM = []
        self.gt_NL = []
        self.predicted_classes = []
        # Define the list of all possible classes
        # GS means Gleason score and has one to one correpondance with ISUP, GS_1= 0 means NC, GS_1 (3+3 = 6) means ISUP =1
        self.classes = [0, 1, 2, 3, 4, 5]
        self.class_mapping = {
            0: 'NC',
            1: 'ISUP_1',
            2: 'ISUP_2',
            3: 'ISUP_3',
            4: 'ISUP_4',
            5: 'ISUP_5'
        }


    # END __init()__

    def loadList_pickle(self, file_name):
        '''
        The file name is provided and path is known i.e output folder.
        It will return a list of lists. each sublist contains fullpath of img/mask files related to a single WSI.
        '''
        output_folder = project.output_dir

        # Load the list object using pickle
        file_path = os.path.join(output_folder, f"{file_name}")
        with open(file_path, "rb") as file:
            list_object = pickle.load(file)

        #print(f'length of loaded pickled list ={len(list_object)}')
        #print(f'type of contents of the loaded pickled list ={type(list_object[0])}')

        return list_object

    def extract_dataFrame(self, file_paths):
        '''
        Extracts dataframe from the paths listed in a sublist of listofLists.
         Sublist containis paths of all patches related to a single WSI.
        '''
        #print(f'extract_dataFrame(): type of file_paths = {type(file_paths)}')#<class 'list'>
        #print(f'extract_dataFrame(): Length of file_paths = {len(file_paths)}')# e.g. 180
        #print(f'extract_dataFrame(): Type of first element of file_paths = {type(file_paths[0])}')#<class 'str'>
        #print(f'extract_dataFrame(): Length of first element of file_paths = {len(file_paths[0])}')#e.g 100
        #print(f'extract_dataFrame(): first element of file_paths = {file_paths[0]}')#/home/senadmin/MAB/Datasets/SICAPv2/myMasks/16B0023614_Block_Region_3_22_3_xini_14850_yini_22437.png

        data = []
        for file_path in file_paths:
            # Extract the filename from the file path
            filename = os.path.basename(file_path)
            #print(f'file_path ={file_path}, filename = {filename}')

            # Extract slide number
            #match = re.search(r'(/home/senadmin/MAB/Datasets/SICAPv2/images/)(\d+)_Block_Region_(\d+_\d+_\d+)_xini_(\d+)_yini_(\d+).jpg', file_path)
            slide_number = re.match(r'^([^_]+)', filename).group(1)
            #print(f'extract_dataFrame(): slide_number = {slide_number}')# 16B0023614

            # Extract block region, xini, and yini using regular expressions
            #match = re.search(r'Block_Region_(\d+_\d+_\d+)_xini_(\d+)_yini_(\d+)', file_path)
            match = re.search(r'Block_Region_(\d+)_(\d+)_(\d+)_xini_(\d+)_yini_(\d+)', file_path)
            #print(f' Block_Region_ = {match.group(1)}')
            #print(f' yini = {match.group(4)}')# e.g. 14850

            if match:
                block_region_parts = match.groups()
                #block_region = match.group(1)
                Region = block_region_parts[0]
                Y = block_region_parts[1]
                X= block_region_parts[2]
                xini = match.group(4)
                yini = match.group(5)
                #print(f'block_region ={block_region}')

                data.append({
                    'File_Path': file_path,
                    'Slide_No.': slide_number,
                    'Block_Region': block_region_parts,#block_region,
                    'Region':block_region_parts[0],
                    'Y': block_region_parts[1],
                    'X': block_region_parts[2],
                    'xini': xini,
                    'yini': yini
                })
            else:
                print("No match found for file path:", file_path)
        # Create a dataframe and populate with data of all patches related to this slide
        df = pd.DataFrame(data)
        #print('Data frame extracted by extract_dataFrame() method of class WSI_DS ')
        #print(df.head(5))

        return df
    #############################################
    #def update_csv_WSI_DS(self, primary_S, secondary_S, isup_grade_S, slide_No, primary_S_prop, secondary_S_prop, isup_grade_S_prop):
    def update_csv_WSI_DS(self, returnTuple, return_Dict=None):
        '''
        A method of class WSI_DS.
        two files will be updated.
        patch csv: will be updated only in the case of segmentation
        slide scv: This will be updated in both seg and classification cases
        '''
        if self.newClassificationInference:
            slide_No = returnTuple[0]
            isup_grade_S_prop = returnTuple[1]

            BagOfPatches_predicted = return_Dict['BagOfPatches_predicted']
            percent_gleson3_NL = return_Dict['percent_gleson3_NL']
            percent_gleson4_NL = return_Dict['percent_gleson4_NL']
            percent_gleson5_NL = return_Dict['percent_gleson5_NL']
            Slide_primary_BM = return_Dict['Slide_primary_BM']
            Slide_Secondary_BM = return_Dict['Slide_Secondary_BM']
            Slide_ISUP_BM = return_Dict['Slide_ISUP_BM']
            Slide_primary_prop_NL = return_Dict['Slide_primary_prop_NL']
            Slide_Secondary_prop_NL = return_Dict['Slide_Secondary_prop_NL']
            Slide_ISUP_prop_NL = return_Dict['Slide_ISUP_prop_NL']
            Slide_ISUP_PredByClassifier = return_Dict['Slide_ISUP_PredByClassifier']
            BM_PredByClassifier_ISUP_agree = return_Dict['BM_PredByClassifier_ISUP_agree']
            propNL_PredByClassifier_agree = return_Dict['propNL_PredByClassifier_agree']
            #print(f'CHECK: {Slide_ISUP_prop_NL},{Slide_Secondary_prop_NL},{Slide_primary_prop_NL},{Slide_ISUP_BM}')
        elif self.newSegmentationInference:
            slide_No=returnTuple[0]
            percent_gleson3=returnTuple[1]
            percent_gleson4=returnTuple[2]
            percent_gleson5=returnTuple[3]
            percent_gleson3_prop=returnTuple[4]
            percent_gleson4_prop=returnTuple[5]
            percent_gleson5_prop=returnTuple[6]
            primary_S=returnTuple[7]
            secondary_S=returnTuple[8]
            isup_grade_S=returnTuple[9]
            primary_S_prop=returnTuple[10]
            secondary_S_prop=returnTuple[11]
            isup_grade_S_prop=returnTuple[12]
        else:
            slide_No = returnTuple[0]
            percent_gleson3 = returnTuple[1]
            percent_gleson4 = returnTuple[2]
            percent_gleson5 = returnTuple[3]
            primary_S = returnTuple[4]
            secondary_S = returnTuple[5]
            isup_grade_S = returnTuple[6]

            Slide_primary_BM=return_Dict['Slide_primary_BM']
            Slide_Secondary_BM=return_Dict['Slide_Secondary_BM']
            Slide_ISUP_BM=return_Dict['Slide_ISUP_BM']
            Primary_agree=return_Dict['Primary_agree']
            Secondary_agree=return_Dict['Secondary_agree']
            slide_agree=return_Dict['slide_agree']
            BagOfLabels = [return_Dict["slide_BOW_ISUP_Prop_dict"]['0'],
                           return_Dict["slide_BOW_ISUP_Prop_dict"]['1'],
                           return_Dict["slide_BOW_ISUP_Prop_dict"]['2'],
                           return_Dict["slide_BOW_ISUP_Prop_dict"]['3'],
                           return_Dict["slide_BOW_ISUP_Prop_dict"]['4'],
                           return_Dict["slide_BOW_ISUP_Prop_dict"]['5'],
                           ]

        output_folder = project.output_dir

        # Load the classification csv file
        file_path_patch = os.path.join(output_folder, f"{self.csv_filename_patch}")
        file_path_slide = os.path.join(output_folder, f"{self.csv_filename_slide}")

        if self.newClassificationInference:
            # in case of classification we don not need patch file we are using actually one.
            # You can create a new df that contains slide No., patch No and patch ISUP  grade
            # compare file names of df

            pass
        elif self.newSegmentationInference:
            # Read the CSV file into a DataFrame
            df_patch = pd.read_csv(file_path_patch)

            # Define the given Slide_No and the new value
            given_slide_no = slide_No
            print(f'slide_No to be looked for in csv for final slide update. = {given_slide_no}')# 0 Why

            # Replace values in the corresponding column based on Slide_No
            '''
            column_to_replace = 'percent_gleson3'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = percent_gleson3
            column_to_replace = 'percent_gleson4'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = percent_gleson4
            column_to_replace = 'percent_gleson5'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = percent_gleson5
            '''
            column_to_replace = 'Slide_primary'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = primary_S
            column_to_replace = 'Slide_Secondary'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = secondary_S
            # Replace values in the corresponding column based on Slide_No
            column_to_replace = 'Slide_ISUP'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = isup_grade_S
            ####################
            # Replace values in the corresponding column based on Slide_No
            '''
            column_to_replace = 'percent_gleson3_prop'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = percent_gleson3_prop
            column_to_replace = 'percent_gleson4_prop'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = percent_gleson4_prop
            column_to_replace = 'percent_gleson5_prop'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = percent_gleson5_prop
            '''
            column_to_replace = 'Slide_p_prop'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = primary_S
            column_to_replace = 'Slide_S_prop'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = secondary_S
            column_to_replace = 'Slide_ISUP_prop'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = isup_grade_S

            print(f'Segmentation case: I am going to write patch data in  = {file_path_patch}')
            # Save the modified DataFrame back to the CSV file
            df_patch.to_csv(file_path_patch, index=False)
        else:
            # Labeling case
            # I think no need to add slide data in patch data csv...........
            '''# Read the CSV file into a DataFrame
            df_patch = pd.read_csv(file_path_patch)

            # Define the given Slide_No and the new value
            given_slide_no = slide_No
            print(f'slide_No to be looked for in csv for final slide update. = {given_slide_no}')  # 0 Why

            # Replace values in the corresponding column based on Slide_No
            column_to_replace = 'Slide_primary'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = primary_S
             column_to_replace = 'Slide_Secondary'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = secondary_S
            # Replace values in the corresponding column based on Slide_No
            column_to_replace = 'Slide_ISUP'
            df_patch.loc[df_patch['Slide_No.'] == given_slide_no, column_to_replace] = isup_grade_S
            ####################
            # Replace values in the corresponding column based on Slide_No

            # Save the modified DataFrame back to the CSV file
            print(f'Labeling case: I am going to write modified patch data in = {file_path_patch}')
            df_patch.to_csv(file_path_patch, index=False)
            '''
            pass
        ###################### Handle slide_only csv file ###############
        # data row for a slide
        if self.newClassificationInference:
            data_slide = [self.S_No, slide_No, BagOfPatches_predicted,
                          percent_gleson3_NL, percent_gleson4_NL, percent_gleson5_NL,
                          Slide_primary_BM, Slide_Secondary_BM, Slide_ISUP_BM,
                          Slide_primary_prop_NL, Slide_Secondary_prop_NL, Slide_ISUP_prop_NL,
                          Slide_ISUP_PredByClassifier,
                          BM_PredByClassifier_ISUP_agree, propNL_PredByClassifier_agree
                          ]
        elif self.newSegmentationInference:
            data_slide = [self.S_No, slide_No,
                          percent_gleson3,percent_gleson4,percent_gleson5,
                          percent_gleson3_prop,percent_gleson4_prop,percent_gleson5_prop,
                          primary_S, secondary_S, isup_grade_S,
                          primary_S_prop, secondary_S_prop, isup_grade_S_prop
                          ]
        else:
            data_slide = [self.S_No, slide_No, BagOfLabels,
                          percent_gleson3, percent_gleson4, percent_gleson5,
                          Slide_primary_BM, Slide_Secondary_BM, Slide_ISUP_BM,
                          primary_S, secondary_S, isup_grade_S,
                          Primary_agree, Secondary_agree, slide_agree,
                          ]
            print(f'CHECK: {return_Dict["slide_No"]} = {slide_No}')
            print(f'CHECK: {isup_grade_S} = {return_Dict["isup_grade_S"]}')

        self.S_No = self.S_No+1

        print(f'I am going to write Slide data in = {file_path_slide}')
        #print(f'data to be written in slideClassification csv file: {data_slide}')
        with open(file_path_slide, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_slide)


    #############################################
    def label_slides(self):
        '''
        This is a method of class WSI_DS.
        This method use list of dataframes and returns a csv file containing slide level GG labels
        along with GG labels of patches
         input masks of datset
        '''
        # full path of csv file that will contain all slides classification data after we have calculated it.
        # We are extracting new labels from masks:'patch_slideClassification'# output and 'slideClassification'# output
        csv_filename_fullpath_patch = project.output_dir.joinpath(self.csv_filename_patch)
        csv_filename_fullpath_slide = project.output_dir.joinpath(self.csv_filename_slide)
        print(f'csv_filename_fullpath_patch = {csv_filename_fullpath_patch}')
        print(f'csv_filename_fullpath_slide = {csv_filename_fullpath_slide}')

        # Open the csv file and write col headings.
        with open(csv_filename_fullpath_patch, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write header of the patch Classification csv file.
            writer.writerow(self.headers_patch)

        with open(csv_filename_fullpath_slide, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write header of the Slide Classification csv file.
            writer.writerow(self.headers_slide)

        # Pick each df from the list and use that df to make WSI object to assign GG to patches and subsequently the Slide GG
        for WSI_No, slide_df in enumerate(self.df_list):
            '''
            WSI_No: index of slide_df returned by enumerator 
            slide_df: data frame containing patches info related to a WSI
            '''
            # print(f'Length of slide_df from df_list = {len(slide_df)}')
            # make object of WSI and call its method to classify the slide.
            mySlide = WSI(slide_dataframe=slide_df,
                          WSI_labels_df=self.WSI_labels_df,
                          S_No=self.S_No,
                          csv_filename_fullpath_patch=csv_filename_fullpath_patch,  # input file
                          newClassificationInference=False,
                          csv_filename_fullpath_slide=csv_filename_fullpath_slide,  # output file
                          )  # chk slide_No

            # Extract data from all patches related to the current WSI and write in patch and Slide csvs
            returnTuple, return_Dict = mySlide.labelSlide()

            self.S_No = self.S_No + 1  # slide counter, to be written in csv file
            # print(f'self.S_No = {self.S_No}')

            # BOW prep for KNN WSI classifier
            #print(f'return_Dict["slide_BOW_ISUP_Prop_dict"] = {return_Dict["slide_BOW_ISUP_Prop_dict"]}')

            self.BagOfWords_ISUP[return_Dict["slide_No"]] = return_Dict['slide_BOW_ISUP_Prop_dict']
            self.BagOfWords_GS[return_Dict["slide_No"]] = return_Dict['slide_BOW_GS_Prop_dict']
            self.BagOfWords_GS_Label[return_Dict["slide_No"]] = return_Dict['slide_GS_Prop_dict']
            self.BagOfWords_ISUP_Label[return_Dict["slide_No"]] = return_Dict['isup_grade_S']
            # Add a new row using append method

            print(f'slide_No = {return_Dict["slide_No"]}')
            print(f'BOW Labels for slide No. {return_Dict["slide_No"]} = {self.BagOfWords_ISUP[return_Dict["slide_No"]]}')

            ISUP0 = return_Dict["slide_BOW_ISUP_Prop_dict"]['0']
            ISUP1 = return_Dict["slide_BOW_ISUP_Prop_dict"]['1']
            ISUP2 = return_Dict["slide_BOW_ISUP_Prop_dict"]['2']
            ISUP3 = return_Dict["slide_BOW_ISUP_Prop_dict"]['3']
            ISUP4 = return_Dict["slide_BOW_ISUP_Prop_dict"]['4']
            ISUP5 = return_Dict["slide_BOW_ISUP_Prop_dict"]['5']

            GS0= return_Dict['slide_BOW_GS_Prop_dict']['0+0']
            GS1 = return_Dict['slide_BOW_GS_Prop_dict']['3+3']
            GS2 = return_Dict['slide_BOW_GS_Prop_dict']['3+4']
            GS3 = return_Dict['slide_BOW_GS_Prop_dict']['3+5']
            GS4 = return_Dict['slide_BOW_GS_Prop_dict']['4+3']
            GS5 = return_Dict['slide_BOW_GS_Prop_dict']['4+4']
            GS6 = return_Dict['slide_BOW_GS_Prop_dict']['4+5']
            GS7 = return_Dict['slide_BOW_GS_Prop_dict']['5+3']
            GS8 = return_Dict['slide_BOW_GS_Prop_dict']['5+4']
            GS9 = return_Dict['slide_BOW_GS_Prop_dict']['5+5']

            new_row = {'Slide_No.': return_Dict["slide_No"],
                       'slide_BOW_ISUP_NL': return_Dict["slide_BOW_ISUP_Prop_dict"],
                       'ISUP0' : ISUP0,
                       'ISUP1' : ISUP1,
                       'ISUP2' : ISUP2,
                       'ISUP3' : ISUP3,
                       'ISUP4' : ISUP4,
                       'ISUP5' : ISUP5,
                       'slide_BOW_GS_NL': return_Dict['slide_BOW_GS_Prop_dict'],
                       'GS0' : GS0,
                       'GS1' : GS1,
                       'GS2' : GS2,
                       'GS3' : GS3,
                       'GS4' : GS4,
                       'GS5' : GS5,
                       'GS6' : GS6,
                       'GS7' : GS7,
                       'GS8' : GS8,
                       'GS9' : GS9,
                       'slide_GS_NL': return_Dict['slide_GS_Prop_dict'],
                       'slide_isup_NL': return_Dict['isup_grade_S'],
                       'slide_GS_BM': '+'.join([str(return_Dict['Slide_primary_BM']), str(return_Dict['Slide_Secondary_BM'])]),
                       'slide_isup_BM': return_Dict['Slide_ISUP_BM'],
                       }
            print(f'new_row = {new_row}')
            wronglyLabeledSlide_df_new_row = {'Slide_No.': return_Dict["slide_No"],
                                                'percent_gleson3':return_Dict['percent_gleson3'],
                                                'percent_gleson4':return_Dict['percent_gleson4'],
                                                'percent_gleson5':return_Dict['percent_gleson5'],
                                                'slide_GS_NL': return_Dict['slide_GS_Prop_dict'],
                                                'slide_isup_NL': return_Dict['isup_grade_S'],
                                                'slide_GS_BM': '+'.join([str(return_Dict['Slide_primary_BM']), str(return_Dict['Slide_Secondary_BM'])]),
                                                'slide_isup_BM': return_Dict['Slide_ISUP_BM'],
                                                'slide_BOW_ISUP_NL': return_Dict["slide_BOW_ISUP_Prop_dict"],
                                                }
            # Note all wrongly predicted slides and save as pickle file at the end.
            if return_Dict['Slide_ISUP_BM'] != return_Dict['isup_grade_S']:
                self.wronglyLabeledSlide_df.loc[len(self.wronglyLabeledSlide_df)] = wronglyLabeledSlide_df_new_row

            #Inserting the  new row
            self.BagOfWords_df.loc[len(self.BagOfWords_df)] = new_row
            # Reset the index
            self.BagOfWords_df = self.BagOfWords_df.reset_index(drop=True)
            #self.BagOfWords_df = self.BagOfWords_df.append(new_row, ignore_index=True)

            # All patches of relevant slide have been classified and at the end using their data slide is also classified
            # now update slide info in the csv file already populated for relevant patches
            self.update_csv_WSI_DS(returnTuple, return_Dict)

        # Open Slide csv here, count agreement and percentage, and rewrite slide csv
        output_folder = project.output_dir
        # Load the classification csv file
        file_path_slide = os.path.join(output_folder, f"{self.csv_filename_slide}")
        # Read the CSV file into a DataFrame
        df_slide = pd.read_csv(file_path_slide)
        # Convert the slide_agree column to integers
        df_slide['Primary_agree'] = df_slide['Primary_agree'].astype(int)
        df_slide['Secondary_agree'] = df_slide['Secondary_agree'].astype(int)
        df_slide['slide_agree'] = df_slide['slide_agree'].astype(int)
        # Calculate the number of 1s in the slide_agree column
        prim_ones = df_slide['Primary_agree'].sum()
        second_ones = df_slide['Secondary_agree'].sum()
        ISUP_ones = df_slide['slide_agree'].sum()

        # Calculate the percentage of agreement
        total_rows = len(df_slide)
        percentage_agreement_prim = round((prim_ones / total_rows) * 100, significantDigits)
        percentage_agreement_second = round((second_ones / total_rows) * 100, significantDigits)
        percentage_agreement_ISUP = round((ISUP_ones / total_rows) * 100, significantDigits)

        print(f'Labeling:I am going to write Stats in Slide data in = {file_path_slide}')
        # print(f'data to be written in slideClassification csv file: {data_slide}')
        with open(file_path_slide, 'a', newline='') as file:
            file.write(f'Number of 1s in Primary_agree: {prim_ones}\n')
            file.write(f'Number of 1s in Secondary_agree: {second_ones}\n')
            file.write(f'Number of 1s in slide_agree: {ISUP_ones}\n')
            file.write(f'\nPercentage agreement in prime GP: {percentage_agreement_prim:.2f}%\n')
            file.write(f'Percentage agreement in secondary GP: {percentage_agreement_second:.2f}%\n')
            file.write(f'Percentage agreement in slide ISUP: {percentage_agreement_ISUP:.2f}%\n')

        # Save BOWs to be used in KNN algo for WSI classification
        file_path = os.path.join(output_folder, "BagOfWords_ISUP.pickle")
        with open(file_path, "wb") as file:
            pickle.dump(self.BagOfWords_ISUP, file)
            file.close()
        file_path = os.path.join(output_folder, "BagOfWords_GS.pickle")
        with open(file_path, "wb") as file:
            pickle.dump(self.BagOfWords_GS, file)
            file.close()
        file_path = os.path.join(output_folder, "BagOfWords_GS_Label.pickle")
        with open(file_path, "wb") as file:
            pickle.dump(self.BagOfWords_GS_Label, file)
            file.close()
        file_path = os.path.join(output_folder, "BagOfWords_ISUP_Label.pickle")
        with open(file_path, "wb") as file:
            pickle.dump(self.BagOfWords_ISUP_Label, file)
            file.close()
        file_path = os.path.join(output_folder, "BagOfWords_df.pickle")
        with open(file_path, "wb") as file:
            pickle.dump(self.BagOfWords_df, file)
            file.close()
        file_path = os.path.join(output_folder, "wronglyLabeledSlide_df.pickle")
        with open(file_path, "wb") as file:
            pickle.dump(self.wronglyLabeledSlide_df, file)
            file.close()


    def classify_slides(self):
        '''
        This is a method of class WSI_DS.
        This method use list of dataframes and returns a csv file containing slide level GG
        along with GG grades of patches

        BM File: comparison_results_OrigClass_OrigMask.csv. This file contains WSI level primary, secondary
        and ISUP Labels of original excel file and extracted labels from masks

        SICAPv2 Extracted Lables from original masks: multclass_Classification_labels, This file should also include
        patch level G3,G4, G5, this file was generated using extractClassLabels.py
        '''

        # Open the csv file and write col headings.
        if self.newClassificationInference:
            # input file: self.infer_Classi_patch_file_name, is Full path of csv file that will contain all
            # patche_inference data provided by classification model in module main_classification.py.
            print(f'Read classification inference from = {self.infer_Classi_patch_file_name}')

            # output file. Full path of csv file that will contain all slides classification data
            # after we have calculated it.
            csv_filename_fullpath_slide = project.output_dir.joinpath(self.csv_filename_slide)
            print(f'Write WSI level classification inference to = {csv_filename_fullpath_slide}')

            with open(csv_filename_fullpath_slide, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write header of the Slide Classification csv file.
                writer.writerow(self.headers_slide)

            # Pick each df from the list and use that df to make WSI object to assign GG to patches and subsequently the Slide GG
            for WSI_No, slide_df in enumerate(self.df_list):
                '''
                WSI_No: index of slide_df returned by enumerator 
                slide_df: data frame containing patches info related to a WSI
                '''
                # print(f'Length of slide_df from df_list = {len(slide_df)}')
                # make object of WSI and call its method to classify the slide.
                mySlide = WSI(slide_dataframe=slide_df,
                              WSI_labels_df=self.WSI_labels_df,
                              S_No=self.S_No,
                              csv_filename_fullpath_patch=self.infer_Classi_patch_file_name,  # input file
                              newClassificationInference=self.newClassificationInference, # bool
                              csv_filename_fullpath_slide=csv_filename_fullpath_slide,  # output file
                              KNNClassifier_obj = self.KNNClassifier_obj,
                              featureType=self.featureType,
                              useKNNClassifier= self.useKNNClassifier,
                              augmentDataset = self.augmentDataset,
                              #reducedBow = self.reducedBow,
                              )  # chk slide_No

                # primary_S, secondary_S, isup_grade_S, slide_No, primary_S_prop, secondary_S_prop, isup_grade_S_prop = mySlide.classifySlide()
                returnTuple, return_Dict = mySlide.classifySlide()

                # ** Data for Confusion Matrix formation *** #
                self.predicted_classes.append(return_Dict['Slide_ISUP_predicted'])
                self.gt_BM.append(return_Dict['Slide_ISUP_BM'])
                self.gt_NL.append(return_Dict['Slide_ISUP_prop_NL'])

                #for key in return_Dict:
                #    print(key)

                wronglyPredictedSlide_df_new_row = {'Slide_No.': return_Dict["slide_No"],
                                                    'BagOfPatches_predicted':return_Dict["BagOfPatches_predicted"],
                                                    'percent_gleson3': return_Dict['percent_gleson3_NL'],
                                                    'percent_gleson4': return_Dict['percent_gleson4_NL'],
                                                    'percent_gleson5': return_Dict['percent_gleson5_NL'],
                                                    'slide_GS_NL': '+'.join([str(return_Dict['Slide_primary_prop_NL']),
                                                                           str(return_Dict['Slide_Secondary_prop_NL'])]),
                                                    'slide_ISUP_NL': return_Dict['Slide_ISUP_prop_NL'],
                                                    'slide_GS_BM': '+'.join([str(return_Dict['Slide_primary_BM']),
                                                                           str(return_Dict['Slide_Secondary_BM'])]),
                                                    'slide_ISUP_BM': return_Dict['Slide_ISUP_BM'],
                                                    'slide_ISUP_Predicted':return_Dict['Slide_ISUP_PredByClassifier'],
                                                    }
                # DEBUG
                print(f'slide_No = {return_Dict["slide_No"]}')
                if return_Dict["slide_No"] == '16B0022914':
                    print(f'slide_No = {return_Dict["slide_No"]}')
                    print(f'Slide_ISUP_PredByClassifier={return_Dict["Slide_ISUP_PredByClassifier"]}')
                    print(f'Slide_ISUP_BM={return_Dict["Slide_ISUP_BM"]}')
                    print(f'Slide_ISUP_prop_NL={return_Dict["Slide_ISUP_prop_NL"]}')

                # Note all wrongly predicted slides and save as pickle file at the end.
                if ((return_Dict['Slide_ISUP_BM'] != return_Dict['Slide_ISUP_PredByClassifier']) or
                        (return_Dict['Slide_ISUP_prop_NL'] != return_Dict['Slide_ISUP_PredByClassifier'])):
                    self.wronglyPredictedSlide_df.loc[len(self.wronglyPredictedSlide_df)] = wronglyPredictedSlide_df_new_row


                self.S_No = self.S_No + 1  # slide counter, to be written in csv file
                # print(f'self.S_No = {self.S_No}')

                # All patches of relevant slide have been classified and at the end using their data slide is also classified
                # now update slide info in the csv file already populated for relevant patches
                self.update_csv_WSI_DS(returnTuple, return_Dict)

            # Final Report Lines at the end of SLIDE Inference file: WSI_ISUP_Predictions_MultiClass_classification_prop.csv
            # Open Slide csv here, count agreement and percentage, and rewrite slide csv
            output_folder = project.output_dir
            # Load the classification csv file
            file_path_slide = os.path.join(output_folder, f"{self.csv_filename_slide}")
            # Read the CSV file into a DataFrame
            df_slide = pd.read_csv(file_path_slide)

            print(f'Length of df_slide ={len(df_slide)}')

            # Convert the slide_agree column to integers
            df_slide['BM_PredByClassifier_ISUP_agree'] = df_slide['BM_PredByClassifier_ISUP_agree'].astype(int)
            df_slide['propNL_PredByClassifier_agree'] = df_slide['propNL_PredByClassifier_agree'].astype(int)
            # Calculate the number of 1s in the slide_agree column
            BM_PredByClassifier_ISUP_agree_ones = df_slide['BM_PredByClassifier_ISUP_agree'].sum()
            propNL_PredByClassifier_agree_ones = df_slide['propNL_PredByClassifier_agree'].sum()

            # Calculate the percentage of agreement
            total_rows = len(df_slide)
            percentage_agreement_BM_prop = round((BM_PredByClassifier_ISUP_agree_ones / total_rows) * 100, significantDigits)
            percentage_agreement_NL_prop = round((propNL_PredByClassifier_agree_ones / total_rows) * 100, significantDigits)

            print(f'New Classification:I am going to write Stats in Slide data in = {file_path_slide}')
            # print(f'data to be written in slideClassification csv file: {data_slide}')
            with open(file_path_slide, 'a', newline='') as file:
                file.write(f'\nNumber of 1s in BM_PredByClassifier_ISUP_agree_ones: {BM_PredByClassifier_ISUP_agree_ones}\n')
                file.write(f'Number of 1s in propNL_PredByClassifier_agree_ones: {propNL_PredByClassifier_agree_ones}\n')
                file.write(f'\nPercentage agreement in BM vs Propoaed: {percentage_agreement_BM_prop:.2f}%\n')
                file.write(f'Percentage agreement in NL vs Propoaed: {percentage_agreement_NL_prop:.2f}%\n')

            file_path1 = os.path.join(output_folder, "wronglyPredictedSlide_df.pickle")
            file_path2 = os.path.join(output_folder, "wronglyPredictedSlide_csv")
            with open(file_path1, "wb") as file:
                pickle.dump(self.wronglyPredictedSlide_df, file)
                file.close()
            # Also save as csv file
            self.wronglyPredictedSlide_df.to_csv(file_path2, index=False)


        elif self.newSegmentationInference: # make it in future segmentation inference
            # input file. Full path of csv file that will contain all patches segmentation inference data
            # provided by segmentation model in module main *.py.
            # Work in progress################

            # full path of csv file that will contain all slides classification data after we have calculated it.
            # We are extracting new labels from masks:'patch_slideClassification'# output and 'slideClassification'# output
            csv_filename_fullpath_patch = project.output_dir.joinpath(self.csv_filename_patch)
            csv_filename_fullpath_slide = project.output_dir.joinpath(self.csv_filename_slide)
            print(f'csv_filename_fullpath_patch = {csv_filename_fullpath_patch}')
            print(f'csv_filename_fullpath_slide = {csv_filename_fullpath_slide}')

            with open(csv_filename_fullpath_patch, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write header of the Slide Classification csv file.
                writer.writerow(self.headers_patch)
                # Open the csv file and write col headings.
            with open(csv_filename_fullpath_slide, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write header of the Slide Classification csv file.
                writer.writerow(self.headers_slide)

            # Pick each df from the list and use that df to make WSI object to assign GG to patches and subsequently the Slide GG
            for WSI_No, slide_df in enumerate(self.df_list):
                '''
                WSI_No: index of slide_df returned by enumerator 
                slide_df: data frame containing patches info related to a WSI
                '''
                #print(f'Length of slide_df from df_list = {len(slide_df)}')
                # make object of WSI and call its method to classify the slide.
                mySlide = WSI(slide_dataframe=slide_df,
                              WSI_labels_df = self.WSI_labels_df,
                              S_No=self.S_No,
                              csv_filename_fullpath_patch=csv_filename_fullpath_patch,#input file
                              newClassificationInference=False,
                              csv_filename_fullpath_slide = csv_filename_fullpath_slide,# output file
                              )# chk slide_No

                #primary_S, secondary_S, isup_grade_S, slide_No, primary_S_prop, secondary_S_prop, isup_grade_S_prop = mySlide.classifySlide()
                returnTuple, return_Dict = mySlide.classifySlide(WSI_labels_df= self.WSI_labels_df)

                self.predicted_classes.append(return_Dict['isup_grade_S_prop'])
                self.ground_truth.append(return_Dict['gt_fromSICAPv2_classification'])


                self.S_No = self.S_No + 1  # slide counter, to be written in csv file
                #print(f'self.S_No = {self.S_No}')

                # All patches of relevant slide have been classified and at the end using their data slide is also classified
                # now update slide info in the csv file already populated for relevant patches
                self.update_csv_WSI_DS(returnTuple)

        #return updated_classification_csv

    ######################################

    def makeConfusionMatrix(self, ground_truth, predicted_classes, classes):
        #print(f'ground_truth length = {len(ground_truth)}')
        #print(f'predicted_classes length = {len(predicted_classes)}')

        #print(f'ground_truth length = {ground_truth}')
        #print(f'predicted_classes length = {predicted_classes}')
        #print(f'classes = {classes}')

        # Compute the confusion matrix
        #confusion_mat = confusion_matrix(ground_truth_labels, predicted_labels, labels=classes)

        gt_list = []
        pred_list = []

        for key in ground_truth:
            gt_list.append(ground_truth[key])
            pred_list.append(predicted_classes[key])

        # Compute the confusion matrix
        confusion_mat = confusion_matrix(ground_truth, predicted_classes, labels=classes)

        # Print the confusion matrix
        '''
        Returns : ndarray of shape (n_classes, n_classes) Confusion matrix whose i-th row and j-th column entry 
        indicates the number of samples with true label being i-th class and predicted label being j-th class.
        '''
        print("Confusion Matrix:")
        print(confusion_mat)

        # Calculate accuracy
        accuracy = np.diag(confusion_mat).sum() / confusion_mat.sum()

        # Calculate recall (sensitivity) using macro average
        recall = recall_score(gt_list, pred_list, average='macro')

        # Calculate specificity using macro average
        specificity = (np.diag(confusion_mat).sum()-np.diag(confusion_mat)).sum()/(confusion_mat.sum()-np.diag(confusion_mat).sum())

        # Calculate precision (positive predictive value) using macro average
        precision = precision_score(gt_list, pred_list, average='macro')

        # Calculate specificity using macro average, Specificity = TN / (TN + FP)
        specificity = []
        for i in range(confusion_mat.shape[0]):
            row = np.delete(confusion_mat, i, axis=0)
            col = np.delete(row, i, axis=1)
            specificity.append(row.sum() / (row.sum() + col.sum()))
        specificity = np.mean(specificity)

        kappa_score = cohen_kappa_score(gt_list, pred_list, weights=None)
        linear_kappa_score = cohen_kappa_score(gt_list, pred_list, weights="linear")
        quadratic_kappa_score = cohen_kappa_score(gt_list, pred_list, weights="quadratic")
        print(f'Cohen kappa:{kappa_score}, linear_kappa_score: {linear_kappa_score}, quadratic_kappa_score : {quadratic_kappa_score}')
        print("Accuracy:", accuracy)
        print("Recall (Sensitivity):", recall)
        print("Specificity:", specificity)
        print("Precision (Positive Predictive Value):", precision)


#class WSI(WSI_DS):
class WSI():
    def __init__(self,
                 slide_dataframe,
                 WSI_labels_df,# BM slide data
                 S_No,
                 csv_filename_fullpath_patch, #
                 csv_filename_fullpath_slide,
                 newClassificationInference=False,
                 useKNNClassifier= 0,
                 augmentDataset = 0,
                 #reducedBow = 0, # 0: include NC in BOW
                 KNNClassifier_obj=None,
                 featureType = None,
                 SVMClassifier_obj = None,
                 ):
        #super(WSI_DS, self).__init__()
        self.featureType = featureType
        #self.reducedBow = reducedBow # 0: use NC in BOW
        self.slide_dataframe = slide_dataframe# This should contain all images/masks related to a single WSI
        self.WSI_labels_df = WSI_labels_df
        self.KNNClassifier_obj = KNNClassifier_obj# Use your algo or KNN or both for WSI classification
        self.SVMClassifier_obj = SVMClassifier_obj#
        self.augmentDataset = augmentDataset # augment Data set at runtime or not
        self.useKNNClassifier=useKNNClassifier # 0 means do not use KNNClassifier, 1: use KNN , 2: use KNN with your own algo
        self.S_No = S_No
        self.slide_No = '0'
        self.patch_No = 0  # patch ini slide (dataframe of slide)
        self.csv_filename_fullpath_patch = csv_filename_fullpath_patch
        self.csv_filename_fullpath_slide = csv_filename_fullpath_slide
        self.GG3_pixels = 0
        self.GG4_pixels = 0
        self.GG5_pixels = 0
        self.totalpixels = 0
        self.GG3_pixels_prop = 0
        self.GG4_pixels_prop = 0
        self.GG5_pixels_prop = 0
        self.totalpixels_prop = 0
        self.significantDigits = 4
        self.newClassificationInference = newClassificationInference
        # All patches_data related to a single slide will go to a single file
        self.headers_patch = ['S.No.', 'patch_path', 'Slide_No.', '%G3','%G4','%G5', 'Infered_Patch_ISUP']
        self.return_Dict = OrderedDict()
    def labelSlide(self):
        '''
        This is the method of class WSI.
        Assign labels to patches and slides based on masks of dataset
        '''
        # histogram  for this slide []
        #slide_BOW_ISUP_BM_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        #slide_BOW_GS_BM_dict = {"0+0": 0, "3+3": 0, "3+4": 0, "3+5": 0, "4+3": 0, "4+4": 0, "4+5": 0, "5+3": 0, "5+4": 0, "5+5": 0}
        slide_BOW_ISUP_Prop_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        slide_BOW_GS_Prop_dict = {"0+0": 0, "3+3": 0, "3+4": 0, "3+5": 0, "4+3": 0, "4+4": 0, "4+5": 0, "5+3": 0, "5+4": 0, "5+5": 0}


        df_obj = self.slide_dataframe  # This should contain all images/masks related to a single WSI
        print(f'Length of df_obj which belong to a slide = {len(df_obj)}')
        # print(f'This df should contain info about input source of inference={df_obj.columns}') #Yes
        # columns:['File_Path', 'Slide_No.', 'Block_Region', 'Region', 'Y', 'X', 'xini','yini'],

        ######################################
        # Sort df_obj in ascending order and rearrange all other columns accordingly
        sorted_df = df_obj.sort_values(by=['Region', 'Y', 'X'], ascending=True)
        print(f'Length of sorted_df which belong to a slide = {len(sorted_df)}')  # CHECK: Its length must equal len(df_obj)
        # This mach patches against this slide
        # print(f'First 5 rows of df related to a sublist i.e one slide(WSI) ={sorted_df.head(5)}')  #
        # print(f'File path of First rows of df related to one WSI ={sorted_df.loc[0,"File_Path"]}')  #

        # get unique regions, x and y coordinates
        unique_regions = sorted_df['Region'].unique()
        unique_patches = sorted_df['File_Path'].unique()
        print(f'Length of unique_patches = {len(unique_patches)}')  # CHECK: This mus equal len(sorted_df)
        unique_slide_No = sorted_df['Slide_No.'].unique()[0]
        self.slide_No = unique_slide_No
        # print(f'Unique slide_No = {unique_slide_No}, type = {type(unique_slide_No)}')#<class 'numpy.ndarray'>
        # print(f'Unique regions = {unique_regions}, type = {type(unique_regions)}')  # <class 'numpy.ndarray'>
        # print(f'Unique patches = {len(unique_patches)}, type = {type(unique_patches)}')# type = <class 'numpy.ndarray'>, each element is full path str
        # print(f'Unique Y = {unique_Y}, type = {type(unique_Y)}')#<class 'numpy.ndarray'>, however each element is a number but type is string
        # print(f'Ubique X = {unique_X}, type = {type(unique_X)}')# type = <class 'numpy.ndarray'>

        for patch_no, (idx, row) in enumerate(sorted_df.iterrows()):
            # print(f'row of sorted_df type = {type(row)}')#<class 'pandas.core.series.Series'>

            # Extract data from this row  i.e related to one patch
            # self.slide_No= row['Slide_No.']
            file_path = row['File_Path']

            # Print the relevant values for each row
            # print(f'Index of this loop= {patch_no}, and row in dataframe ={idx}') # What is this why its value is 114
            #print('Labeling case, mask File path: File_Path:', file_path)

            # get relevant image first
            patch_np = io.imread(file_path)

            # *********************************************************************
            # Use whole ndarray of the patch.
            # This is very important function call. Here patch is assigned ISUP.
            # This is tricky you should try different estimation schemes to assign ISUP grade to patch.
            # This can change performance of segmentation inference
            # *********************************************************************
            patch_dict = utils.SICAP_mask_statistics(patch_np)

            # Now accumulate stats of all patches related to a single WSI to be used to assign WSI level ISUP grade
            # Used full patch for calculatio
            self.GG3_pixels += int(patch_dict['num_1_pixels'])
            self.GG4_pixels += int(patch_dict['num_2_pixels'])
            self.GG5_pixels += int(patch_dict['num_3_pixels'])
            #self.totalpixels += int(patch_dict['num_pixels'])
            self.totalpixels += int(patch_dict['num_1_pixels'])+int(patch_dict['num_2_pixels'])+int(patch_dict['num_3_pixels'])

            #  CHECK ststs of any given slide **********
            if self.slide_No == '':
                print(f'self.GG3_pixels ={self.GG3_pixels}')
                print(f'self.GG4_pixels ={self.GG4_pixels}')
                print(f'self.GG5_pixels ={self.GG5_pixels}')
                print(f'self.totalpixels ={self.totalpixels}')


            # data row for a patch
            data_patch = [self.S_No + 1, self.slide_No, patch_no,
                          patch_dict['percent_gleson3'], patch_dict['percent_gleson4'], patch_dict['percent_gleson5'],
                          patch_dict['primary'], patch_dict['secondary'], patch_dict['isup_grade_tile'],
                          ]
            with open(self.csv_filename_fullpath_patch, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data_patch)

            # print(f"Patch {patch_no} just Updated CSV file")
            self.patch_No = self.patch_No + 1

            #Update BOW for KNN classification of WSI
            slide_BOW_ISUP_Prop_dict, slide_BOW_GS_Prop_dict = self.updateBOW(patch_dict,
                                                                              slide_BOW_ISUP_Prop_dict,
                                                                              slide_BOW_GS_Prop_dict)
            #print(f'slide_BOW_ISUP_Prop_dict values = {slide_BOW_ISUP_Prop_dict}')

        ##### Now deal with Slide (WSI) #####
        # Here WSI level ISUP assignment for Labeling case is handled
        # Call here method of slide (WSI) to calc slide GGs and put last entry in csv,
        # infact just modify last entry of last patch of this slide
        if self.totalpixels >0:
            percent_gleson3 = round((self.GG3_pixels / self.totalpixels) * 100, self.significantDigits)
            percent_gleson4 = round((self.GG4_pixels / self.totalpixels) * 100, self.significantDigits)
            percent_gleson5 = round((self.GG5_pixels / self.totalpixels) * 100, self.significantDigits)
        else:
            percent_gleson3 = 0
            percent_gleson4 = 0
            percent_gleson5 = 0

        if self.slide_No == '':
            print(f'percent_gleson3 ={percent_gleson3}')
            print(f'percent_gleson4 ={percent_gleson4}')
            print(f'percent_gleson5 ={percent_gleson5}')

        primary_S, secondary_S, isup_grade_S = utils.slideGrade(self.GG3_pixels,
                                                                self.GG4_pixels,
                                                                self.GG5_pixels,
                                                                self.totalpixels,
                                                                labeling=1,
                                                                prediction=0,
                                                                slideNo= self.slide_No,
                                                                significantDigits= self.significantDigits
                                                                )
        #*****  CHECK Assigned Slide Labels of any given slide **********#
        if self.slide_No == '':
            print(f'primary_S ={primary_S}')
            print(f'secondary_S ={secondary_S}')
            print(f'isup_grade_S ={isup_grade_S}')

        # **** Now SOME DATA FOR BENCH MARKING *******#
        # Extract following data from self.WSI_labels_df
        # Filter the DataFrame based on the slide_id
        filtered_df = self.WSI_labels_df[self.WSI_labels_df['slide_id'] == self.slide_No]
        # Extract the values of Gleason_primary, Gleason_secondary
        Slide_primary_BM = filtered_df['Gleason_primary'].item()
        Slide_Secondary_BM = filtered_df['Gleason_secondary'].item()
        Slide_ISUP_BM = self.assignISUPtoGT(Slide_primary_BM, Slide_Secondary_BM)

        if Slide_primary_BM == primary_S:
            Primary_agree=1
        else:
            Primary_agree = 0

        if Slide_Secondary_BM==secondary_S:
            Secondary_agree=1
        else:
            Secondary_agree = 0

        if Slide_ISUP_BM==isup_grade_S:
            slide_agree=1
        else:
            slide_agree = 0

        returnTuple = (self.slide_No,
                       percent_gleson3, percent_gleson4, percent_gleson5,
                       primary_S, secondary_S, isup_grade_S,
                       Slide_primary_BM, Slide_Secondary_BM,Slide_ISUP_BM,
                       slide_BOW_ISUP_Prop_dict,slide_BOW_GS_Prop_dict,
                       '+'.join([str(primary_S), str(secondary_S)])
                       )

        self.return_Dict['slide_No'] = self.slide_No
        self.return_Dict['percent_gleson3'] = percent_gleson3
        self.return_Dict['percent_gleson4'] = percent_gleson4
        self.return_Dict['percent_gleson5'] = percent_gleson5
        self.return_Dict['primary_S'] = primary_S
        self.return_Dict['secondary_S'] = secondary_S
        self.return_Dict['isup_grade_S'] = isup_grade_S
        self.return_Dict['Slide_primary_BM'] = Slide_primary_BM
        self.return_Dict['Slide_Secondary_BM'] = Slide_Secondary_BM
        self.return_Dict['Slide_ISUP_BM'] = Slide_ISUP_BM
        self.return_Dict['Primary_agree'] = Primary_agree
        self.return_Dict['Secondary_agree'] = Secondary_agree
        self.return_Dict['slide_agree'] = slide_agree
        self.return_Dict['slide_BOW_ISUP_Prop_dict'] = slide_BOW_ISUP_Prop_dict
        self.return_Dict['slide_BOW_GS_Prop_dict'] = slide_BOW_GS_Prop_dict
        self.return_Dict['slide_GS_Prop_dict'] = '+'.join([str(primary_S), str(secondary_S)])
        print(f'primary_S={self.return_Dict["primary_S"]} + secondary_S= {self.return_Dict["secondary_S"]}')
        print(f'slide_BOW_ISUP_Prop_dict=\n{self.return_Dict["slide_BOW_ISUP_Prop_dict"]}')
        print(f'slide_BOW_GS_Prop_dict=\n{self.return_Dict["slide_BOW_GS_Prop_dict"]}')

        # return WSI level (Gleason grades):GG3, GG4, GG5 and ISUP (gleason group, based on Gleason scores)
        return returnTuple, self.return_Dict

    def updateBOW(self, patch_dict, slide_BOW_ISUP_Prop_dict, slide_BOW_GS_Prop_dict):
        '''
        Get data for each patch related to a WSI and update BOW
        '''
        primary = patch_dict['primary']
        secondary= patch_dict['secondary']
        isup_grade_tile = patch_dict['isup_grade_tile']
        #print(f'[updateBOW()]: primary {primary}, secondary {secondary}, and isup_grade_tile {isup_grade_tile}')

        if primary == 0 and secondary== 0:
            slide_BOW_ISUP_Prop_dict["0"] = slide_BOW_ISUP_Prop_dict["0"] + 1
            slide_BOW_GS_Prop_dict["0+0"] =slide_BOW_GS_Prop_dict["0+0"] + 1
        elif primary == 3 and secondary==3:
            slide_BOW_ISUP_Prop_dict["1"] = slide_BOW_ISUP_Prop_dict["1"] + 1
            slide_BOW_GS_Prop_dict["3+3"] =slide_BOW_GS_Prop_dict["3+3"] + 1
        elif primary == 3 and secondary==4:
            slide_BOW_ISUP_Prop_dict["2"] = slide_BOW_ISUP_Prop_dict["2"] + 1
            slide_BOW_GS_Prop_dict["3+4"] =slide_BOW_GS_Prop_dict["3+4"] + 1
        elif primary == 3 and secondary == 5:
            slide_BOW_ISUP_Prop_dict["4"] = slide_BOW_ISUP_Prop_dict["4"] + 1
            slide_BOW_GS_Prop_dict["3+5"] = slide_BOW_GS_Prop_dict["3+5"] + 1
        elif primary == 4 and secondary== 3:
            slide_BOW_ISUP_Prop_dict["3"] = slide_BOW_ISUP_Prop_dict["3"] + 1
            slide_BOW_GS_Prop_dict["4+3"] =slide_BOW_GS_Prop_dict["4+3"] + 1
        elif primary == 4 and secondary==4:
            slide_BOW_ISUP_Prop_dict["4"] = slide_BOW_ISUP_Prop_dict["4"] + 1
            slide_BOW_GS_Prop_dict["4+4"] =slide_BOW_GS_Prop_dict["4+4"] + 1
        elif primary == 4 and secondary== 5:
            slide_BOW_ISUP_Prop_dict["5"] = slide_BOW_ISUP_Prop_dict["5"] + 1
            slide_BOW_GS_Prop_dict["4+5"] =slide_BOW_GS_Prop_dict["4+5"] + 1
        elif primary == 5 and secondary==3:
            slide_BOW_ISUP_Prop_dict["4"] = slide_BOW_ISUP_Prop_dict["4"] + 1
            slide_BOW_GS_Prop_dict["5+3"] =slide_BOW_GS_Prop_dict["5+3"] + 1
        elif primary == 5 and secondary==4:
            slide_BOW_ISUP_Prop_dict["5"] = slide_BOW_ISUP_Prop_dict["5"] + 1
            slide_BOW_GS_Prop_dict["5+4"] =slide_BOW_GS_Prop_dict["5+4"] + 1
        elif primary == 5 and secondary == 5:
            slide_BOW_ISUP_Prop_dict["5"] = slide_BOW_ISUP_Prop_dict["5"] + 1
            slide_BOW_GS_Prop_dict["5+5"] = slide_BOW_GS_Prop_dict["5+5"] + 1
        else:
            print(f'There must be some error in updateBOW()')

        # Iterate over the dictionary and print key-value pairs
        #for key, value in slide_BOW_ISUP_Prop_dict.items():
        #    print(key, "->", value)
        #for key, value in slide_BOW_GS_Prop_dict.items():
        #    print(key, "->", value)

        return slide_BOW_ISUP_Prop_dict, slide_BOW_GS_Prop_dict
    def classifySlide(self):
        '''
        This is the method of class WSI.
        If inference is from segmentation model, this method calls patch class object to calculate primary and secondary
        patch grades, and subsequently when all mask patches related to single WSI are analysed, on the basis of
        accumulated grades slide primary and secondary grades are calculated and slide level ISUP grade is assigned.

        If inference is from classification model, this method extracts patch file names info from dataframe relevant to single
        slide, from csv file of patch inferences looks patch ISUP grade and on ther basis of these ISUP grades calculates
        WSI level ISUP grade.
        '''

        df_obj = self.slide_dataframe# This should contain all images/masks related to a single WSI
        print(f'Length of df_obj which belong to a slide = {len(df_obj)}')
        #print(f'This df should contain info about input source of inference={df_obj.columns}') #Yes
        # columns:['File_Path', 'Slide_No.', 'Block_Region', 'Region', 'Y', 'X', 'xini','yini'],

        ######################################
        # Sort df_obj in ascending order and rearrange all other columns accordingly
        sorted_df = df_obj.sort_values(by=['Region', 'Y', 'X'], ascending=True)
        print(f'Length of sorted_df which belong to a slide = {len(sorted_df)}')# CHECK: Its length must equal len(df_obj)
        # This much patches against this slide
        #print(f'First 5 rows of df related to a sublist i.e one slide(WSI) ={sorted_df.head(5)}')  #
        #print(f'File path of First rows of df related to one WSI ={sorted_df.loc[0,"File_Path"]}')  #

        # get unique regions, x and y coordinates
        unique_regions  = sorted_df['Region'].unique()
        unique_patches= sorted_df['File_Path'].unique()
        print(f'Length of unique_patches = {len(unique_patches)}')# CHECK: This mus equal len(sorted_df)
        unique_Y = sorted_df['Y'].unique()
        unique_X = sorted_df['X'].unique()
        unique_slide_No = sorted_df['Slide_No.'].unique()[0]

        # Now we have slide no to be graded, this slide number will be looked into inference_made_by_model and BM_file.
        self.slide_No = unique_slide_No
        print(f'Unique slide_No = {unique_slide_No}, type = {type(unique_slide_No)}')#<class 'numpy.ndarray'>

        #print(f'Unique regions = {unique_regions}, type = {type(unique_regions)}')  # <class 'numpy.ndarray'>
        #print(f'Unique patches = {len(unique_patches)}, type = {type(unique_patches)}')# type = <class 'numpy.ndarray'>, each element is full path str
        #print(f'Unique Y = {unique_Y}, type = {type(unique_Y)}')#<class 'numpy.ndarray'>, however each element is a number but type is string
        #print(f'Ubique X = {unique_X}, type = {type(unique_X)}')# type = <class 'numpy.ndarray'>

        if self.newClassificationInference:
            '''
            here in this loop all patches of a single slide are dealt, so we can make a single CSV per WSI. As we have
            only one file for inference from the classification model, we can extract info from that and make as many
            CSv as there are WSIs, those could be used later on for further processing like slide level classification.
            '''
            ################################## Patch files per slide ####NOT required Yet
            # Write data to a csv about all patches related to this slide.
            '''self.headers_patch = ['S.No.', 'patch_path', 'Slide_No.', 'Infered_Patch_ISUP', 'gt_Patch_ISUP']

            # We dont have , '%G3','%G4','%G5' in inference file for classification. We just have Prim Grade and secnd grade

            # path for WSI related csv will be different from source like inferenceSplit_by_pandas.py
            # This file will contain all patches data related to a single slide extracted from a single patch level inferencefile
            inference_dir_path_modified = os.path.join(inference_dir_path,'patchStich')
            if not os.path.exists(inference_dir_path_modified):
                # Create a new directory because it does not exist
                os.makedirs(inference_dir_path_modified)
                print(f'Made a new dir for inferences = {inference_dir_path_modified}')

            # following file will contain all patches info relevant to single WSI
            patch_inferences_for_slide = os.path.join(inference_dir_path_modified, self.slide_No)
            with open(patch_inferences_for_slide, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write header of the Slide Classification csv file.
                # ['S.No.', 'patch_path', 'Slide_No.', 'Infered_Patch_ISUP', 'gt_Patch_ISUP']
                writer.writerow(self.headers_patch)
            '''
            pass
            ###############################################################
        else:
            # Segmentation based inference is used
            # every patch is an img file. A region may contain many patches.
            # Enumerate over the sorted DataFrame, while getting whole row
            # Extract data from all patches of a single slide

            for patch_no, (idx, row) in enumerate(sorted_df.iterrows()):
                #print(f'row of sorted_df type = {type(row)}')#<class 'pandas.core.series.Series'>

                # Extract data from this row  i.e related to one patch
                #self.slide_No= row['Slide_No.']
                file_path = row['File_Path']
                region = row['Region']
                y_value = row['Y']  # value returned by row['Y'] is ndarray of strings
                x_value = row['X']
                xini = row['xini']
                yini = row['yini']

                # Print the relevant values for each row
                #print(f'Index of this loop= {patch_no}, and row in dataframe ={idx}') # What is this why its value is 114
                print('New Segmentation Inference: File_Path:', file_path)
                #print(f'Y:{y_value}, of type={type(y_value)}')#Y:0, of type=<class 'str'>
                #print('X:', x_value)
                #print(f'xini = {xini}, yini = {yini}')

                # Make a patch object, and call relevant methods to classify it
                patch_location = ( int(xini), int(yini))

                # set coordinates of current patch in its object
                patch_obj = Patch(patch_location, self.patch_No, self.filename)

                # Find and set adjacent patches coordinates in the patch object
                patch_obj.calculate_adjacent_coordinates(sorted_df, region)# region is not really required

                if not patch_obj.has_patch_right and not patch_obj.has_patch_above:
                    which_part = 4
                    #print(f"No patch right and No patch above. Use whole array.")
                elif not patch_obj.has_patch_right and patch_obj.has_patch_above:
                    which_part = 2
                    #print(f"No patch right but there is a patch above. Use 1 and 2 quarters.")
                elif patch_obj.has_patch_right and not patch_obj.has_patch_above:
                    which_part = 3
                    #print(f'Patch right but not below. Use 1 and 3 quarters.')
                else:
                    which_part = 1
                    #print(f"Both Right and Above patchs present. Use 1 quarter only.")
                    above_coordinates = patch_obj.above_coordinates
                    right_coordinates = patch_obj.right_coordinates
                    #print(f'patch_no {patch_no}has coordinates:{patch_obj.location},'
                    #      f'above_coordinates={above_coordinates},'
                    #      f'right_coordinates={right_coordinates}')

                # get relevant image first
                patch_np = io.imread(file_path)

                # Use whole ndarray of the patch.
                # This is very important function call. Here patch is assigned ISUP.
                # This is tricky you should try different estimation schemes to assign ISUP grade to patch.
                # This can change performance of segmentation inference
                patch_dict = utils.SICAP_mask_statistics(patch_np)

                ########################################
                self.headers_patch = ['S.No.', 'patch_path', 'Slide_No.', '%G3', '%G4', '%G5', 'Patch_ISUP']
                patch_data=(patch_no, file_path, self.slide_No,
                            patch_dict['percent_gleson3'],
                            patch_dict['percent_gleson4'],
                            patch_dict['percent_gleson5'],
                            sorted_df['Slide_ISUP']
                            # ISUP from from inference file___________________
                            )
                with open(os.path.join(inference_dir_path, self.slide_No), 'a', newline='') as file:
                    writer = csv.writer(file)
                    # Write header of the Slide Classification csv file.
                    writer.writerow(patch_data)
                #########################################

                # Now accumulate stats of all patches related to a single WSI to be used to assign WSI level ISUP grade
                # Used full patch for calculation
                self.GG3_pixels += int(patch_dict['num_1_pixels'])
                self.GG4_pixels += int(patch_dict['num_2_pixels'])
                self.GG5_pixels += int(patch_dict['num_3_pixels'])
                #self.totalpixels += int(patch_dict['num_pixels'])
                self.totalpixels += int(patch_dict['num_1_pixels'])+int(patch_dict['num_2_pixels'])+int(patch_dict['num_3_pixels'])

                # Use a portion of patch_np according to the position of patch in a region
                patch_dict_prop = utils.SICAP_mask_statistics(patch_np, which_part)
                self.GG3_pixels_prop += int(patch_dict['num_1_pixels'])
                self.GG4_pixels_prop += int(patch_dict['num_2_pixels'])
                self.GG5_pixels_prop += int(patch_dict['num_3_pixels'])
                #self.totalpixels_prop += int(patch_dict['num_pixels'])
                self.totalpixels_prop += int(patch_dict['num_1_pixels'])+int(patch_dict['num_2_pixels'])+int(patch_dict['num_3_pixels'])

                # data row for a patch
                data_patch = [self.S_No+1, self.slide_No, patch_no,
                              patch_dict['percent_gleson3'], patch_dict['percent_gleson4'], patch_dict['percent_gleson5'],
                              patch_dict_prop['percent_gleson3'], patch_dict_prop['percent_gleson4'], patch_dict_prop['percent_gleson5'],
                              patch_dict['primary'],patch_dict['secondary'], patch_dict['isup_grade_tile'],
                              patch_dict_prop['primary'], patch_dict_prop['secondary'], patch_dict_prop['isup_grade_tile'],
                              'Slide primary','Slide Secondary', 'Slide ISUP'
                              ]
                self.update_csv_WSI_DS(data_patch)
                #print(f"Patch {patch_no} just Updated CSV file")
                self.patch_No = self.patch_No+1

        ##### Now deal with Slide (WSI) #####
        if self.newClassificationInference:
            # Now load your classification inference csv from the given path and extract ISUP of all patchs which given
            # in the dataframe for this WSI.
            # Then get WSI ISUP grade using some method like majority vote.
            #print(f'sorted_df = {sorted_df.head(2)}')
            #print(f'sorted_df have file path  = {sorted_df.loc[0,"File_Path"]}')

            #self.filename #csv_filename_fullpath_patch , input file

            # Read the classification patch_inference CSV file and create a DataFrame.
            # This DF contains patch inferences of all the dataset NOT JUST this slide under consideration
            # we will have to extract all the relevant patches from it for grading the slide under consideration
            class_inf_df = pd.read_csv(self.csv_filename_fullpath_patch)
            print(f'Length of input file that contain inference of classification = {len(class_inf_df)}')
            #print(f'class_inf_df have file path  = {class_inf_df.loc[0, "img_file_name"]}')

            # Perform some processing on the columns of df1=contains file paths or df2=contains all inferences
            # Extract the patch_file_name without extension using apply() FROM df containing all files relevant to one WSI
            sorted_df['Img_File_Name'] = sorted_df['File_Path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

            # Extract the file name without extension FROM classification infernce file containing all Dataset files
            class_inf_df['Img_File_Name'] = class_inf_df['img_file_name'].apply(lambda x: os.path.splitext(x)[0])

            # Both sources have different no of files so no one to one correspondance so be careful. ***********
            #print(f'no one to one correspondance: {sorted_df.loc[0,"Img_File_Name"]}={class_inf_df.loc[0,"Img_File_Name"]}')

            # Merge the dataframes based on the processed columns
            merged_df = pd.merge(sorted_df, class_inf_df, left_on='Img_File_Name', right_on='Img_File_Name', how='inner')
            '''
            By setting how='inner', only the matching rows where the values in the specified columns are equal will be 
            included in the merged dataframe. Rows with non-matching values will be excluded.
            It's important to note that the resulting merged dataframe will contain only the columns from both 
            dataframes that are involved in the merge operation. If you want to include additional columns from either 
            dataframe, you can modify the code accordingly or use other parameters provided by pd.merge()
            '''
            #print(f'CHECK:Length of merged_df = {len(merged_df)} = {len(sorted_df)}')
            #print(f'Colums of merged_df are = {merged_df.columns}')
            # ['File_Path', 'Slide_No.', 'Block_Region', 'Region', 'Y', 'X', 'xini',
            #        'yini', 'Img_File_Name', 'S.No.', 'img_file_name', 'Slide_ISUP',
            #        'gt_fromSICAPv2_classification', 'Agreement']
            #print(f'merged_df entries = {merged_df["Img_File_Name"]}')

            # Drop the 'col_to_drop' column using drop()
            merged_df = merged_df.drop('Block_Region', axis=1)
            merged_df = merged_df.drop('Region', axis=1)
            merged_df = merged_df.drop('Y', axis=1)
            merged_df = merged_df.drop('X', axis=1)
            merged_df = merged_df.drop('xini', axis=1)
            merged_df = merged_df.drop('yini', axis=1)
            merged_df = merged_df.drop('S.No.', axis=1)

            #print(f'First row of merged_df ={merged_df.loc[0]}')
            # Count the number of ones in the 'Agreement' column, this was fill during inferce by model
            count_ones = merged_df['Agreement'].sum()

            # Get the unique values and their frequencies in the 'Slide_ISUP' column, e.g grade values may be 0,1,2,3,4,5
            ############Get frequencies of all classes even if one is not returned #############
            value_counts = merged_df['Slide_ISUP'].value_counts()
            #print(f'Frequencies of all classes in this slide before re indexing ={value_counts}')
            # Reindex the value_counts series with all six classes and fill missing values with 0
            value_counts = value_counts.reindex(range(6), fill_value=0)

            #print(f'Frequencies of all classes in this slide ={value_counts}')
            ###############################
            print(f'Accuray of patch inference = {round(((count_ones/len(sorted_df))*100),2)}, for Slide_No ={merged_df.loc[0,"Slide_No."]}')
            #print(f'Slide No.= {merged_df.loc[0,"Slide_No."]}, have value_counts ={value_counts}')

            print(f'Lenght of BM dataframe= {len(self.WSI_labels_df)}')

            # Assign ISUP grade to WSI slide on the basis of patch_ISUP inferences
            isup_grade_S_prop = self.assignISUPtoWSI(slide_no=self.slide_No,
                                                     patch_ISUP_Count=value_counts,# BagOfPatches_predicted
                                                     num_Of_Patches=len(sorted_df),
                                                     agreement_Count=count_ones,
                                                     WSI_labels_df=self.WSI_labels_df,# gt
                                                     )

            # Find the matching row based on the slide number, WSI_labels_df we got from 'comparison_results_OrigClass_OrigMask.csv'
            #print(f'WSI_labels_df = {self.WSI_labels_df.columns}')
            # col heads:['S.No.', 'Slide_No.', 'percent_gleson3', 'percent_gleson4',
            #        'percent_gleson5', 'Slide_primary_BM', 'Slide_Secondary_BM',
            #        'Slide_ISUP_BM', 'Slide_primary_prop', 'Slide_Secondary_prop',
            #        'Slide_ISUP_prop', 'Primary_agree', 'Secondary_agree', 'slide_agree'],

            # Filter the DataFrame based on the slide number
            #filtered_row = self.WSI_labels_df.loc[self.WSI_labels_df['Slide_No.'] == self.slide_No]
            #print(f'filtered_row= {filtered_row}')
            '''#matching_row = WSI_labels_df[WSI_labels_df['img_file_name'].str.startswith(self.slide_No + '_')]
            matching_row = self.WSI_labels_df[self.WSI_labels_df['Slide_No.'].str.startswith(self.slide_No)]

            # Retrieve the value in gt_fromSICAPv2_classification column from the matching row
            if not matching_row.empty:
                Slide_primary_prop = matching_row['Slide_primary_prop'].values[0]
                Slide_Secondary_prop = matching_row['Slide_Secondary_prop'].values[0]
                print(f'gt_fromSICAPv2_classification {Slide_primary_prop}, {Slide_Secondary_prop}')
            '''
            #gt_fromSICAPv2_classification = self.assignISUPtoGT(Slide_primary_prop,Slide_Secondary_prop)# already
            # present in slideClassification.csv

            returnTuple = (self.slide_No, isup_grade_S_prop )#isup_grade_S_prop is actually predicted

            # **** Now SOME DATA FOR BENCH MARKING *******#
            # Extract following data from self.WSI_labels_df
            # Filter the DataFrame based on the slide_id
            BM_Classi_df = self.WSI_labels_df[self.WSI_labels_df['Slide_No.'] == self.slide_No]
            # Extract the values of patient_id, Gleason_primary, Gleason_secondary

            #print(f'CHECK: Slide_ISUP_prop = {type(BM_Classi_df["Slide_ISUP_prop"].item())}')# float

            if isup_grade_S_prop == BM_Classi_df['Slide_ISUP_BM'].item():
                BM_PredByClassifier_ISUP_agree = 1
            else:
                BM_PredByClassifier_ISUP_agree = 0

            if isup_grade_S_prop == BM_Classi_df['Slide_ISUP_prop'].item():
                propNL_PredByClassifier_agree = 1
            else:
                propNL_PredByClassifier_agree = 0

            self.return_Dict['slide_No'] = self.slide_No
            #self.return_Dict['BagOfPatches_predicted'] = value_counts # done where classified
            self.return_Dict['Slide_ISUP_predicted'] = isup_grade_S_prop # formerly prop
            #self.return_Dict['gt_fromSICAPv2_classification'] = gt_fromSICAPv2_classification
            self.return_Dict['percent_gleson3_NL'] = BM_Classi_df['percent_gleson3'].item()
            self.return_Dict['percent_gleson4_NL'] = BM_Classi_df['percent_gleson4'].item()
            self.return_Dict['percent_gleson5_NL'] = BM_Classi_df['percent_gleson5'].item()
            self.return_Dict['Slide_primary_BM'] = int(BM_Classi_df['Slide_primary_BM'].item())
            self.return_Dict['Slide_Secondary_BM'] = int(BM_Classi_df['Slide_Secondary_BM'].item())
            self.return_Dict['Slide_ISUP_BM'] = int(BM_Classi_df['Slide_ISUP_BM'].item())
            self.return_Dict['Slide_primary_prop_NL'] = int(BM_Classi_df['Slide_primary_prop'].item())
            self.return_Dict['Slide_Secondary_prop_NL'] = int(BM_Classi_df['Slide_Secondary_prop'].item())
            self.return_Dict['Slide_ISUP_prop_NL'] = int(BM_Classi_df['Slide_ISUP_prop'].item())
            self.return_Dict['Slide_ISUP_PredByClassifier'] =isup_grade_S_prop
            self.return_Dict['BM_PredByClassifier_ISUP_agree'] = BM_PredByClassifier_ISUP_agree
            self.return_Dict['propNL_PredByClassifier_agree'] = propNL_PredByClassifier_agree
        else:
            # Here WSI level ISUP assignment for segmentation case is handled
            # Call here method of slide (WSI) to calc slide GGs and put last entry in csv,
            # infact just modify last entry of last patch of this slide
            primary_S, secondary_S, isup_grade_S = utils.slideGrade(self.GG3_pixels,
                                                                    self.GG4_pixels,
                                                                    self.GG5_pixels,
                                                                    #labeling=0,
                                                                    #prediction=1,
                                                                    #slideNo=self.slide_No,
                                                                    #significantDigits=self.significantDigits
                                                                    )
            primary_S_prop, secondary_S_prop, isup_grade_S_prop = utils.slideGrade(self.GG3_pixels_prop,
                                                                                   self.GG4_pixels_prop,
                                                                                   self.GG5_pixels_prop
                                                                                   )
            percent_gleson3 = round((self.GG3_pixels / self.totalpixels) * 100,self.significantDigits)
            percent_gleson4 = round((self.GG4_pixels / self.totalpixels) * 100,self.significantDigits)
            percent_gleson5 = round((self.GG5_pixels / self.totalpixels) * 100,self.significantDigits)

            percent_gleson3_prop = round((self.GG3_pixels_prop / self.totalpixels_prop) * 100,self.significantDigits)
            percent_gleson4_prop = round((self.GG4_pixels_prop / self.totalpixels_prop) * 100,self.significantDigits)
            percent_gleson5_prop = round((self.GG5_pixels_prop / self.totalpixels_prop) * 100,self.significantDigits)

            returnTuple = (self.slide_No,
                           percent_gleson3, percent_gleson4, percent_gleson5,
                           percent_gleson3_prop, percent_gleson4_prop, percent_gleson5_prop,
                           primary_S, secondary_S, isup_grade_S,
                           primary_S_prop, secondary_S_prop, isup_grade_S_prop
                           )
            ################## Collect info for Confusion matrix and
            # Find the matching row based on the slide number, WSI_labels_df we got from 'comparison_results_OrigClass_OrigMask.csv'
            matching_row = self.WSI_labels_df[self.WSI_labels_df['img_file_name'].str.startswith(self.slide_No + '_')]

            # Retrieve the value in gt_fromSICAPv2_classification column from the matching row
            if not matching_row.empty:
                gt_fromSICAPv2_classification = matching_row['gt_fromSICAPv2_classification'].values[0]
                print(f'gt_fromSICAPv2_classification {gt_fromSICAPv2_classification}')

            returnTuple = (self.slide_No, isup_grade_S_prop)
            self.return_Dict['slide_No'] = self.slide_No
            self.return_Dict['isup_grade_S_prop'] = isup_grade_S_prop
            self.return_Dict['gt_fromSICAPv2_classification'] = gt_fromSICAPv2_classification
            self.return_Dict['isup_grade_S'] = isup_grade_S


        # return WSI level (Gleason grades):GG3, GG4, GG5 and ISUP (gleason group, based on Gleason scores)
        return returnTuple, self.return_Dict

    def update_WSI_csv(self, data):
        '''Method of WSI(), not just one WSI not whole DS'''
        print(f'The path of csv file used for patch data update in classification csv file = {self.csv_filename_fullpath_patch}')
        # Take user input
        user_input = input(f'I am going to write data in csv_filename_fullpath_patch {self.csv_filename_fullpath_patch}, Enter "y" to continue or "N" to exit: ')
        # Check user input
        if user_input == 'y':
            with open(self.filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        else:
            # Exit the program
            print("Exiting the program.")
            exit()

    def assignISUPtoWSI(self, slide_no, patch_ISUP_Count, num_Of_Patches, agreement_Count,
                        WSI_labels_df):
        '''

        '''

        #print(f'Now assign the ISUP to WSI based on patch ISUPs')
        #print(f'type of patch_ISUP_Count = {type(patch_ISUP_Count)}')#<class 'pandas.core.series.Series'>
        print(f'Contents of patch_ISUP_Count = {patch_ISUP_Count}, num_Of_Patches = {num_Of_Patches}')

        # You can access the values and their frequencies using the index and values attributes of the Series, respectively. For example:
        #print(f'ISUP frequencies = {patch_ISUP_Count.values}')
        #print(f'unique ISUP values = {patch_ISUP_Count.index}')

        num_of_ISUPS = len(patch_ISUP_Count)
        percentage_agreement = round(((agreement_Count/num_Of_Patches)*100),self.significantDigits)
        print(f'percentage_agreement ={percentage_agreement}')


        '''0+0 or "negative"    0
            3+3(6)      1
            3+4(7)      2
            4+3(7)      3
            4+4(8)      4
            3+5(8)      4
            5+3(8)      4
            4+5(9)      5
            5+4(9)      5
            5+5(10)     5
        '''
        '''for ISUP_grade in patch_ISUP_Count:
            # Access the frequency value by passing ISUP grade as a key
            isup_grade = ISUP_grade
            frequency = patch_ISUP_Count[isup_grade]
            print(f'For isup_grade {isup_grade}, frequency = {frequency}')
        '''
        # Find the maximum value from the 'patch_ISUP_Count' Series
        max_value = patch_ISUP_Count.max() # its frequency
        # Find the ISUP grade with the maximum frequency
        max_grade = patch_ISUP_Count.idxmax() # Its ISUP grade
        print(f'patch_ISUP_Count.max() ={max_value}, patch_ISUP_Count.idxmax()={max_grade}')

        # Retrieve the value in the Slide_ISUP column where Slide_No. matches current slide number
        # Not  assigned  while extracting ????WSI_ISUP_label = WSI_labels_df.loc[WSI_labels_df['Slide_No'] == slide_no, 'Slide_ISUP_prop'].values[0]
        # check  extractClassLabels.py
        WSI_primar_label_BM = WSI_labels_df.loc[WSI_labels_df['Slide_No.'] == slide_no, 'Slide_primary_prop'].values[0]
        WSI_secondary_label_BM = WSI_labels_df.loc[WSI_labels_df['Slide_No.'] == slide_no, 'Slide_Secondary_prop'].values[0]
        WSI_ISUP_BM = self.assignISUPtoGT(WSI_primar_label_BM,WSI_secondary_label_BM )

        NC = patch_ISUP_Count[0]
        GS_1 = patch_ISUP_Count[1]
        GS_2 = patch_ISUP_Count[2]
        GS_3 = patch_ISUP_Count[3]
        GS_4 = patch_ISUP_Count[4]
        GS_5 = patch_ISUP_Count[5]

        percent_NC = round(((NC/num_Of_Patches)*100),2)
        percent_GS_1 = round(((GS_1 / num_Of_Patches) * 100), 2)
        percent_GS_2 = round(((GS_2 / num_Of_Patches) * 100), 2)
        percent_GS_3 = round(((GS_3 / num_Of_Patches) * 100), 2)
        percent_GS_4 = round(((GS_4 / num_Of_Patches) * 100), 2)
        percent_GS_5 = round(((GS_5 / num_Of_Patches) * 100), 2)

        print(f'percent_NC={percent_NC}, percent_GS_1={percent_GS_1}, percent_GS_2={percent_GS_2}')
        print(f'percent_GS_3= {percent_GS_3}, percent_GS_4= {percent_GS_4}, percent_GS_5= {percent_GS_5}')

        ISUP_0_threshold = 90 #%
        ISUP_3_threshold = 10#2
        ISUP_x_threshold = 10#2 # FOR % of patches
        # epsteinCriteria is based on total ISUP count of patches in this WSI it is not % of GP5 in WSI
        # Not exactly epsteinCriteria. GP greater then 5 %, 0.03
        epsteinCriteria = 0.5#5

        G_scores = [percent_NC, percent_GS_1 , percent_GS_2, percent_GS_3 , percent_GS_4 , percent_GS_5 ]
        print(f'G_scores of slide {slide_no} = {G_scores}')
        max_value = max(G_scores)

        first_max_index = G_scores.index(max_value)

        second_max_value = max([num for num in G_scores if num != max_value], default=0)
        second_max_index = G_scores.index(second_max_value)

        third_max_value = max([num for num in G_scores if ((num != max_value) and(num != second_max_value))], default=0)
        third_max_index = G_scores.index(third_max_value)

        fourth_max_value = max([num for num in G_scores if ((num != max_value) and
                                                            (num != second_max_value) and
                                                            (num != third_max_value)
                                                            )], default=0
                               )
        fourth_max_index = G_scores.index(fourth_max_value)

        fifth_max_value = max([num for num in G_scores if ((num != max_value) and
                                                           (num != second_max_value) and
                                                           (num != third_max_value) and
                                                           (num != fourth_max_value)
                                                           )], default=0
                              )
        fifth_max_index = G_scores.index(fifth_max_value)
        print(f'self.useKNNClassifier of WSI(): {self.useKNNClassifier}')
        if self.useKNNClassifier == 1:
            print(f'Use only KNNClassifier for WSI classification')
            if self.featureType == 'ISUP':
                # prepare BOW

                new_ISUP_feature_KNN = [patch_ISUP_Count[0], patch_ISUP_Count[1],
                                        patch_ISUP_Count[2], patch_ISUP_Count[3],
                                        patch_ISUP_Count[4], patch_ISUP_Count[5]]
                self.return_Dict['BagOfPatches_predicted']=new_ISUP_feature_KNN
                pred_label = self.KNNClassifier_obj.Predict([new_ISUP_feature_KNN],
                                                               label=[WSI_ISUP_BM],  # BM
                                                               )
                WSI_ISUP = pred_label[0]
                #if pred_label != WSI_ISUP_BM:
                    # log it into some file
            elif self.featureType == 'GS':
                new_ISUP_feature = []
            else:
                print(f'[useKNNClassifier={self.useKNNClassifier}]: There must be a featureType e.g. ISUP. ')

            if self.augmentDataset == 1:
                self.KNNClassifier_obj.augmentDataset(new_ISUP_features=[new_ISUP_feature],
                                                      new_labels_ISUP=[WSI_ISUP]
                                                      )
                self.KNNClassifier_obj.trainKnnClassifier()
        elif self.useKNNClassifier == 0 or self.useKNNClassifier==2:
            print(f'Use your algo (with(useKNNClassifier=2) or without(useKNNClassifier=0) KNN classification')
            # WSI classiofication
            if first_max_index == 0 and percent_NC >= ISUP_0_threshold:
                WSI_ISUP = 0
            #elif first_max_index != 0:
            #    WSI_ISUP = first_max_index
            else:
                # exclude NC from estimation, now first_max_index will always be 1, and we will not check this
                G_scores = [100, percent_GS_1, percent_GS_2, percent_GS_3, percent_GS_4, percent_GS_5]
                print(f'G_scores of slide {slide_no} = {G_scores}')
                max_value = max(G_scores)
                first_max_index = G_scores.index(max_value)

                second_max_value = max([num for num in G_scores if num != max_value], default=0)
                second_max_index = G_scores.index(second_max_value)

                third_max_value = max([num for num in G_scores if ((num != max_value) and (num != second_max_value))],
                                      default=0)
                third_max_index = G_scores.index(third_max_value)

                fourth_max_value = max([num for num in G_scores if ((num != max_value) and
                                                                    (num != second_max_value) and
                                                                    (num != third_max_value)
                                                                    )], default=0
                                       )
                fourth_max_index = G_scores.index(fourth_max_value)

                fifth_max_value = max([num for num in G_scores if ((num != max_value) and
                                                                   (num != second_max_value) and
                                                                   (num != third_max_value) and
                                                                   (num != fourth_max_value)
                                                                   )], default=0
                                      )
                fifth_max_index = G_scores.index(fifth_max_value)
                sixth_max_value = max([num for num in G_scores if ((num != max_value) and
                                                                   (num != second_max_value) and
                                                                   (num != third_max_value) and
                                                                   (num != fourth_max_value) and
                                                                   (num != fifth_max_value)
                                                                   )], default=0
                                      )
                sixth_max_index = G_scores.index(sixth_max_value)
                print(f'\nSecond MAX Value = {second_max_value} and Second MAX Index = {second_max_index}.')
                print(f'Third MAX Value = {third_max_value} and Third MAX Index = {third_max_index}.')
                print(f'Fourth MAX Value = {fourth_max_value} and Fourth MAX Index = {fourth_max_index}.')
                print(f'Fifth MAX Value = {fifth_max_value} and Fifth MAX Index = {fifth_max_index}.')
                print(f'SIXTH MAX Value = {sixth_max_value} and SIXTH MAX Index = {sixth_max_index}.')

                # In future work also find sixth_max_value and sixth_max_index, to be used in ISUP guidlined directed grading

                #max_val_list = [second_max_value, third_max_value , fourth_max_value, fifth_max_value]
                #max_idx_list = [second_max_index, third_max_index, fourth_max_index, fifth_max_index]
                counter_110000_R =0
                counter_110000_W = 0
                if third_max_value==0 and fourth_max_value==0 and fifth_max_value==0 and sixth_max_value == 0:
                    WSI_ISUP = second_max_index
                    print(f'[assignISUPtoWSI()]: WSI_ISUP {WSI_ISUP} = WSI_ISUP_BM {WSI_ISUP_BM}')
                    '''
                        new_ISUP_feature_KNN = [patch_ISUP_Count[0], patch_ISUP_Count[1],
                                            patch_ISUP_Count[2], patch_ISUP_Count[3],
                                            patch_ISUP_Count[4], patch_ISUP_Count[5]]
                        pred_label = self.KNNClassifier_obj.KNNPredict([new_ISUP_feature_KNN],
                                                                       label=[WSI_ISUP_BM],  # BM
                                                                       )
                        #if pred_label == WSI_ISUP_BM:
                        WSI_ISUP = pred_label[0]
                    elif (1==1):
                    # KnnClassifier beyound this point
                    if self.featureType == 'ISUP':
                        new_ISUP_feature = [patch_ISUP_Count[0], patch_ISUP_Count[1],
                                            patch_ISUP_Count[2], patch_ISUP_Count[3],
                                            patch_ISUP_Count[4], patch_ISUP_Count[5]]
                    elif self.featureType == 'GS':
                        new_ISUP_feature = []
                    else:
                        print(f'There must be a featureType. ')
                    pred_label = self.KNNClassifier_obj.KNNPredict([new_ISUP_feature],
                                                                   label=[WSI_ISUP_BM], #BM
                                                                   )
                    WSI_ISUP = pred_label[0]
                    print(f'[assignISUPtoWSI()]: WSI_ISUP {WSI_ISUP} = WSI_ISUP_BM {WSI_ISUP_BM}')
                    if WSI_ISUP != WSI_ISUP_BM:
                        print(f'[assignISUPtoWSI()]: Wrongly classified feature are {new_ISUP_feature}, and right label is WSI_ISUP {WSI_ISUP_BM}')
                        self.KNNClassifier_obj.augmentDataset(new_ISUP_features=[new_ISUP_feature],
                                                              new_labels_ISUP=[WSI_ISUP_BM])
                        self.KNNClassifier_obj.trainKnnClassifier()
                    '''
                elif (second_max_index==1):#(3+3)-->1
                    # There is no tertiary pattern
                    #if third_max_value != 0 and fourth_max_value == 0 and fifth_max_value == 0:
                    print(f'fourth_max_value ={fourth_max_value} >= fifth_max_value= {fifth_max_value}')
                    if GS_5 > epsteinCriteria:# (4/5+5)-->5
                        #WSI_ISUP = 4  # (3+4)-->2# (3+5)-->4
                        '''if ((0.6 * percent_GS_3) > (0.7 * percent_GS_5)) and (percent_GS_4 > percent_GS_5):
                            WSI_ISUP = 2 # (3+4)-->2
                        elif ((0.6 * percent_GS_3) > (0.9 * percent_GS_5)) and (percent_GS_4 > percent_GS_5):
                            WSI_ISUP = 3 # (4+3)-->3
                        else:
                        '''
                        if(1==1):
                            new_ISUP_feature_KNN = [patch_ISUP_Count[0], patch_ISUP_Count[1],
                                                    patch_ISUP_Count[2], patch_ISUP_Count[3],
                                                    patch_ISUP_Count[4], patch_ISUP_Count[5]]
                            pred_label = self.KNNClassifier_obj.KNNPredict([new_ISUP_feature_KNN],
                                                                           label=[WSI_ISUP_BM],  # BM
                                                                           )
                            WSI_ISUP = pred_label[0]
                    elif  third_max_index == 2 and third_max_value < ISUP_x_threshold:#(3+4)-->2
                        WSI_ISUP = 1
                    elif third_max_index == 2 and third_max_value > ISUP_x_threshold:#(3+4)-->2
                        WSI_ISUP = 2 #(3+4)-->2
                    elif  third_max_index == 3 and third_max_value < ISUP_x_threshold:#(4+3)-->3
                        WSI_ISUP = 1
                    elif third_max_index == 3 and third_max_value > ISUP_x_threshold:#(4+3)-->3
                        WSI_ISUP = 2 #(3+4)-->2
                    elif  third_max_index == 4 and third_max_value < ISUP_x_threshold:#(4+4), (3+5), (5+3)-->4
                        WSI_ISUP = 1
                    elif third_max_index == 4 and third_max_value > ISUP_x_threshold:#(4+4), (3+5), (5+3)-->4
                        WSI_ISUP = 2 #(3+4)-->2  OR (3+5)-->4(low prob)
                    #elif  third_max_index == 5 and (second_max_value/third_max_value) > 5:#(4+5), (5+4), (5+5)-->5
                    elif third_max_index == 5 and (third_max_value) >= ISUP_x_threshold:  # (4+5), (5+4), (5+5)-->5
                        WSI_ISUP = 4#1 #(3+5)-->4 or (4/5+4/5)-->5
                        #WSI_ISUP = 4 if ((1.0 * percent_GS_4) > (0.1 * percent_GS_5)) else 5
                    elif third_max_index == 5 and (third_max_value) < ISUP_x_threshold:#(4+5), (5+4), (5+5)-->5
                        WSI_ISUP = 1
                    else:
                        WSI_ISUP = 1
                elif (second_max_index==2): #(3+4)-->GS_2, (4+3)-->GS_3,
                    #if (third_max_index == 0):  ## this will never happen because we have put 100 in 1st index
                    '''if third_max_index == 1 and third_max_value < ISUP_x_threshold:  # (3+3)-->1
                        WSI_ISUP = 2
                    elif third_max_index == 1 and third_max_value > ISUP_x_threshold:  # (3+3)-->
                        WSI_ISUP = 2
                    '''
                    if GS_5 >epsteinCriteria:
                        if GS_3 > GS_5:
                            WSI_ISUP = 2  # (3+4)-->2 # (3+5),(5+3),(4+4)-->4, (4+5),(5+4),(5+5)-->5
                        elif GS_4 > GS_5:#(3+5),(5+3),(4+4)-->GS_4
                            WSI_ISUP = 4  # (3+5)-->4 # (3+5),(5+3),(4+4)-->4, (4+5),(5+4),(5+5)-->5
                        else:
                            WSI_ISUP = 5
                    elif (third_max_value >ISUP_x_threshold) and (third_max_index > second_max_index):
                        if third_max_index == 5:  # (4+4), (4+5), (5+4)-->5, # (3+4)-->2
                            WSI_ISUP = 4#3# (3+5)-->4
                        elif third_max_index == 4 and (third_max_value ) >= ISUP_3_threshold:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 2 if ((0.7*percent_GS_2)+(0.3*percent_GS_4))>((0.7*percent_GS_4)+(0.3*percent_GS_2)) else 3
                        elif third_max_index == 4 and third_max_value >= epsteinCriteria:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 4
                        if third_max_index == 3 and third_max_value >= ISUP_x_threshold:  # (4+3)-->3
                            WSI_ISUP = 2
                        elif third_max_index == 3 and third_max_value > ISUP_x_threshold:  # (4+3)-->3
                            WSI_ISUP = 2 if ((0.7 * percent_GS_2) + (0.3 * percent_GS_3)) > ((0.7 * percent_GS_3) + (0.3 * percent_GS_2)) else 3
                        else:
                            WSI_ISUP = 2
                    elif (fourth_max_value > ISUP_x_threshold) and (fourth_max_index>second_max_index):
                        if fourth_max_index == 5:  # (4+4), (4+5), (5+4)-->5
                            WSI_ISUP = 4#3# (3+4)-->2 OR (3+5)-->4, (3+4)-->GS_2
                        elif fourth_max_index == 4 and fourth_max_value >= ISUP_3_threshold:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 2 if ((0.7*percent_GS_2)+(0.3*percent_GS_4))>((0.7*percent_GS_4)+(0.3*percent_GS_2)) else 3
                        elif fourth_max_index == 4 and third_max_value >= epsteinCriteria:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 4
                        else:
                            WSI_ISUP = 2
                    elif fifth_max_value > ISUP_x_threshold and (fifth_max_index>second_max_index):
                        if fifth_max_index == 5:  # (4+4), (4+5), (5+4)-->5
                            WSI_ISUP = 4#3# (3+4)-->4
                        elif fifth_max_index == 4 and fifth_max_value >= ISUP_3_threshold:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 2 if ((0.7*percent_GS_2)+(0.3*percent_GS_4))>((0.7*percent_GS_4)+(0.3*percent_GS_2)) else 3
                        elif fifth_max_index == 4 and fifth_max_value >= epsteinCriteria:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 4
                        else:
                            WSI_ISUP = 2
                    elif sixth_max_value > ISUP_x_threshold and (sixth_max_index>second_max_index):
                        if sixth_max_index == 5:  # (4+4), (4+5), (5+4)-->5
                            WSI_ISUP = 4#3# (3+4)-->4
                        elif sixth_max_index == 4 and sixth_max_value >= ISUP_3_threshold:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 2 if ((0.7*percent_GS_2)+(0.3*percent_GS_4))>((0.7*percent_GS_4)+(0.3*percent_GS_2)) else 3
                        elif sixth_max_index == 4 and sixth_max_value >= epsteinCriteria:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 4
                        else:
                            WSI_ISUP = 2
                        '''        
                        if third_max_index == 3 and third_max_value < ISUP_x_threshold:  # (4+3)-->3
                            WSI_ISUP = 2
                        elif third_max_index == 3 and third_max_value > ISUP_x_threshold:  # (4+3)-->3
                            WSI_ISUP = 2 if ((0.7*percent_GS_2)+(0.3*percent_GS_3))>((0.7*percent_GS_3)+(0.3*percent_GS_2)) else 3
                        elif third_max_index == 4 and third_max_value < ISUP_x_threshold:  # (4+4), (3+5), (5+3)-->4
                            WSI_ISUP = 2
                        elif third_max_index == 4 and third_max_value > ISUP_x_threshold:  # (4+4), (3+5), (5+3)-->4
                            WSI_ISUP = 2 if ((0.7*percent_GS_2)+(0.3*percent_GS_4))>((0.7*percent_GS_4)+(0.3*percent_GS_2)) else 4
                        #elif  third_max_index == 5 and (second_max_value/third_max_value) > 5:#(4+5), (5+4), (5+5)-->5
                        elif third_max_index == 5 and (third_max_value) >= epsteinCriteria:  # (4+5), (5+4), (5+5)-->5
                            WSI_ISUP = 2#4#2 # (3+4)-->2 OR (3+5)-->4
                        #elif third_max_index == 5 and (second_max_value/third_max_value) < 5:#(4+5), (5+4), (5+5)-->5
                        elif third_max_index == 5 and (third_max_value) < epsteinCriteria:  # (4+5), (5+4), (5+5)-->5
                            WSI_ISUP = 2#4
                        '''
                    else:
                        WSI_ISUP = 2
                elif second_max_index==3:# (4+3)-->3
                    #if third_max_index == 0:# this will never happen because we have put 100 in 1st index
                    '''if third_max_index == 1 and third_max_value > ISUP_x_threshold:  # (3+3)-->1
                        WSI_ISUP = 3 if (0.8 * percent_GS_3)  > (percent_GS_1) else 2
                    elif third_max_index == 2 and third_max_value < ISUP_x_threshold:  # (3+4)-->2
                        WSI_ISUP = 3
                    elif third_max_index == 2 and third_max_value > ISUP_x_threshold:  # (3+4)-->2
                        WSI_ISUP = 3 if ((0.6 * percent_GS_3) + (0.4 * percent_GS_2)) > ((0.6 * percent_GS_2) + (0.4 * percent_GS_3)) else 2
                    '''
                    if GS_5 >epsteinCriteria:# (4+5), (5+4), (5+5)-->5, # (4+3)-->GS_3 i.e GP4 is already greater
                        if GS_4 >(1*GS_5):
                            WSI_ISUP = 4  # (4+4)-->4 OR (4+5/3)-->4     (4+4),(3+5),(5+3)-->4
                        #elif ((0.6*percent_GS_3)+(0.4*percent_GS_5))>((0.6*percent_GS_5)+(0.4*percent_GS_3)):
                        #    WSI_ISUP = 4  # (4+4)-->4
                        else:
                            values = [3, 4, 5]  # List of values
                            WSI_ISUP = random.choice(values)  # Select a value randomly from the list
                            #WSI_ISUP = 3 #(4+5)-->5,
                        # if (0.4 * percent_GS_3) is higher that means its probbable the 5 is due to 4+5 so G$ is dominant
                        #WSI_ISUP = 4 if (0.8 * percent_GS_3)  > (1.0 * percent_GS_5) else 5
                    elif (third_max_value >epsteinCriteria) and (third_max_index > second_max_index):
                        if third_max_index == 5:  # (4+4), (4+5), (5+4)-->5
                            WSI_ISUP = 5#3# (4+4)-->4 OR (4+5)-->5
                        elif third_max_index == 4 and (third_max_value ) >= ISUP_3_threshold:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 4
                        elif third_max_index == 4 and third_max_value >= epsteinCriteria:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 5
                        #elif third_max_index == 5 and (third_max_value/second_max_value) > epsteinCriteria:#(4+5), (5+4), (5+5)-->5, remember its ratio so nearin 1 mean equal
                        else:
                            WSI_ISUP = 3
                    elif (fourth_max_value > epsteinCriteria) and (fourth_max_index>second_max_index):
                        if fourth_max_index == 5:  # (4+4), (4+5), (5+4)-->5
                            WSI_ISUP = 5#3# (4+4)-->4 OR (4+5)-->5
                        elif fourth_max_index == 4 and fourth_max_value >= ISUP_3_threshold:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 4
                        elif fourth_max_index == 4 and third_max_value >= epsteinCriteria:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 5
                        else:
                            WSI_ISUP = 3
                    elif fifth_max_value > epsteinCriteria and (fifth_max_index>second_max_index):
                        if fifth_max_index == 5:  # (4+4), (4+5), (5+4)-->5
                            WSI_ISUP = 5#3# (4+4)-->4 OR (4+5)-->5
                        elif fifth_max_index == 4 and fifth_max_value >= ISUP_3_threshold:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 4
                        elif fifth_max_index == 4 and fifth_max_value >= epsteinCriteria:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 5
                        else:
                            WSI_ISUP = 3
                    elif sixth_max_value > ISUP_x_threshold and (sixth_max_index>second_max_index):
                        if sixth_max_index == 5:  # (4+4), (4+5), (5+4)-->5
                            WSI_ISUP = 5#3# (4+4)-->4 OR (4+5)-->5
                        elif sixth_max_index == 4 and sixth_max_value >= ISUP_3_threshold:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 4
                        elif sixth_max_index == 4 and sixth_max_value >= epsteinCriteria:  # (4+4), (5+3), (3+5)-->4, 0.03
                            WSI_ISUP = 5
                        else:
                            WSI_ISUP = 3
                    else:
                        WSI_ISUP = 3
                elif second_max_index==4:
                    '''if (third_max_index == 0):  # (4+4), (3+5), (5+3)-->4
                        WSI_ISUP = 4
                    elif third_max_index == 1 and third_max_value < ISUP_x_threshold:  # (3+3)-->1
                        WSI_ISUP = 4
                    elif third_max_index == 1 and third_max_value > ISUP_x_threshold:  # (3+3)-->1
                        WSI_ISUP = 4 if (0.6 * percent_GS_4) > (percent_GS_1) else 3  # this complicated
                    elif third_max_index == 2 and third_max_value < ISUP_x_threshold:  # (3+4)-->2
                        WSI_ISUP = 4
                    elif third_max_index == 2 and third_max_value > ISUP_x_threshold:  # (3+4)-->2
                        WSI_ISUP = 4 if ((0.6 * percent_GS_4) + (0.4 * percent_GS_2)) > ((0.6 * percent_GS_2) + (0.4 * percent_GS_4)) else 3
                    elif third_max_index == 3 and third_max_value < ISUP_x_threshold:  # (4+3)-->3
                        WSI_ISUP = 4
                    elif third_max_index == 3 and third_max_value > ISUP_x_threshold:  # (4+3)-->3
                        WSI_ISUP = 4 if ((0.6 * percent_GS_4) + (0.4 * percent_GS_3)) > ((0.6 * percent_GS_4) + (0.4 * percent_GS_3)) else 2
                    #elif  third_max_index == 5 and (second_max_value/third_max_value) > 5:#(4+5), (5+4), (5+5)-->5
                    '''
                    if GS_5 >epsteinCriteria:#(4+5)-->5 OR (5+5)-->5 OR (5+4)-->5
                        WSI_ISUP = 5  # 3# (3+5)-->4
                    elif third_max_index == 5 and (third_max_value) >= epsteinCriteria:  # (4+5), (5+4), (5+5)-->5
                        WSI_ISUP = 5#4# if second_max_index was because of (4+4)-->4 then (4+4)-->4,(4+5)-->5
                    #elif third_max_index == 5 and (second_max_value/third_max_value) < 5:#(4+5), (5+4), (5+5)-->5, remember its ratio so nearin 1 mean equal
                    elif third_max_index == 5 and (third_max_value) < epsteinCriteria:  # (4+5), (5+4), (5+5)-->5, remember its ratio so nearin 1 mean equal
                        WSI_ISUP = 4#5
                    else:
                        WSI_ISUP = 4
                elif second_max_index==5:#(4=5), (5+40, (5+5) -->5
                    WSI_ISUP = 5
                    '''if third_max_index == 1 and third_max_value < ISUP_x_threshold:  # (3+3)-->1
                        WSI_ISUP = 5
                    elif third_max_index == 1 and third_max_value > ISUP_x_threshold:  # (3+3)-->1
                        WSI_ISUP = 5 if (0.4 * percent_GS_5) > (percent_GS_1) else 4  # this complicated
                    elif third_max_index == 2 and third_max_value < ISUP_x_threshold:  # (3+4)-->2
                        WSI_ISUP = 5
                    elif third_max_index == 2 and third_max_value > ISUP_x_threshold:  # (3+4)-->2
                        WSI_ISUP = 5 if ((0.4 * percent_GS_5) + (0.4 * percent_GS_2)) > ((0.6 * percent_GS_2)) else 4
                    elif third_max_index == 3 and third_max_value < ISUP_x_threshold:  # (4+3)-->3
                        WSI_ISUP = 5
                    elif third_max_index == 3 and third_max_value > ISUP_x_threshold:  # (4+3)-->3
                        WSI_ISUP = 5
                    #elif  third_max_index == 4 and (second_max_value/third_max_value) > 5:#(4+4), (5+3),(3+5)-->4
                    #    WSI_ISUP = 5 # (4+5), (5+4), (5+5)-->5
                    #elif third_max_index == 4 and (second_max_value/third_max_value) < 5:#(4+4), (5+3),(3+5)-->4, remember its ratio so nearin 1 mean equal
                    #    WSI_ISUP = 4
                    else:
                        WSI_ISUP = 5
                    '''
        #print(f'estimated WSI_ISUP ={WSI_ISUP} and WSI_ISUP_label={WSI_ISUP_label}')
        print(f'estimated WSI_ISUP ={WSI_ISUP} and (gt)WSI_primar_label={WSI_primar_label_BM}')
        print(f'estimated WSI_ISUP ={WSI_ISUP} and WSI_secondary_label={WSI_secondary_label_BM}')

        # Calculate gt ISUP grade for this slide from

        return WSI_ISUP

    def assignISUPtoGT(self, primarSlideGrade, secondarySlideGrade):
        '''

        '''
        if primarSlideGrade + secondarySlideGrade == 0:
            gt_ISUP = 0
        elif primarSlideGrade + secondarySlideGrade == 6:
            gt_ISUP = 1
        elif primarSlideGrade + secondarySlideGrade == 7 and primarSlideGrade == 3:
            gt_ISUP = 2
        elif primarSlideGrade + secondarySlideGrade == 7 and primarSlideGrade == 4:
            gt_ISUP = 3
        elif primarSlideGrade + secondarySlideGrade == 8:
            gt_ISUP = 4
        elif primarSlideGrade + secondarySlideGrade == 9 or primarSlideGrade + secondarySlideGrade == 10:
            gt_ISUP = 5
        else:
            print(f'Please check score is not Valid. ISUP grade can not be assigned.')

        return gt_ISUP


class Patch():
    '''
    Finds if there are any adjacent patches

    '''
    def __init__(self, location, patch_No, filename):
        self.location = location
        self.patch_No = patch_No
        self.filename = filename#   chek if used  ASIM
        self.has_patch_above = False
        self.has_patch_below = False
        self.has_patch_left = False
        self.has_patch_right = False

        self.above_coordinates = None
        self.below_coordinates = None
        self.left_coordinates = None
        self.right_coordinates = None

    def calculate_adjacent_coordinates(self, df, region):
        '''
        # df should contain all images/masks related to a single WSI
        '''
        #print(f'\nInside calculate_adjacent_coordinates() of patch class.')
        region_df = df[df['Region'] == region]
        #print(f'The region_df of selected slide and selected region within it ={region_df.head(2)}.')
        if region_df.empty:
            print(f'This region is not present in the dataframe.')
            return None

        # select the first row of the DataFrame region_df and assign it to the variable region_row.
        region_row = region_df.iloc[0]
        #print(f'region_row = {region_row}')
        current_yini_str = region_row['yini']
        current_xini_str = region_row['xini']
        current_yini_int = int(region_row['yini'])
        current_xini_int = int(region_row['xini'])
        #print(f'current_y_str = {current_yini_str}, current_x_str = {current_xini_str}')
        #print(f'set location when initialized, to be compared with every x,y in patch = {self.location}')

        above = df[(df['yini'] == str(current_yini_int + 1024)) & (df['xini'] == current_xini_str)]
        below = df[(df['yini'] == str(current_yini_int - 1024)) & (df['xini'] == current_xini_str)]
        left = df[(df['xini'] == str(current_xini_int - 1024)) & (df['yini'] == current_yini_str)]
        right = df[(df['xini'] == str(current_xini_int + 1024)) & (df['yini'] == current_yini_str)]

        #print(f'above = {current_yini_int + 1024}, below = {current_yini_int - 1024}')
        #print(f'left = {current_xini_int - 1024}, right = {current_xini_int + 1024}')

        if above.empty:
            self.has_patch_above = False
        else:
            above_row = above.iloc[0]
            #print(f'type of above_row = {type(above_row)}')
            self.above_coordinates = (above_row['xini'], above_row['yini'])
            self.has_patch_above = True

        if below.empty:
            self.has_patch_below = False
        else:
            below_row = below.iloc[0]
            #print(f'type of above_row = {type(below_row)}')
            self.below_coordinates = (below_row['xini'], below_row['yini'])
            self.has_patch_below = True
        if left.empty:
            self.has_patch_left = False
        else:
            left_row = left.iloc[0]
            #print(f'type of above_row = {type(left_row)}')
            self.left_coordinates = (left_row['xini'], left_row['yini'])
            self.has_patch_left=True
        if right.empty:
            self.has_patch_right = False
        else:
            right_row = right.iloc[0]
            #print(f'type of above_row = {type(right_row)}')
            self.right_coordinates = (right_row['xini'], right_row['yini'])
            self.has_patch_right=True

        return 1

###############################################

##############################################



#######################################


if __name__ == '__main__':
    # These files should be placed in output folder of the project
    file_name_img = "image_slide_listOflists.pickle"
    file_name_mask = "mask_slide_listOflists.pickle"
    # If segmentation inference is being dealt
    #inference_dir_path= '/home/hpcladmin/MAB/Projects/PyTorch-Deep-Learning/poutyne_Logs/NW_FlexibleNet_DL_SICAPV2_img_NWLs_7_KS_3_BS_4_W_True_A_False_WDyn_False_LC_10_monitor_val_loss_NWDecKer_3_SE_True_RR_16_Reg_No_Regularization_l2_weight_1e-07'
    #inference_dir_path = '/home/hpcladmin/MAB/Projects/PyTorch-Deep-Learning/poutyne_Logs/NW_PFPN_DL_SICAPV2_img_BS_8_W_False_A_False_WDyn_True'
    #inference_dir_path = '/home/hpcladmin/MAB/Projects/PyTorch-Deep-Learning/poutyne_Logs/main_UNet_old_SICAPV2_img_NW_UNet_BS_8_W_False_A_False'
    #inference_dir_path = os.path.join(inference_dir_path, project.inference_dir_name)

    # if classification inference is being dealt
    inference_dir_path = '/home/hpcladmin/MAB/Projects/PyTorch-Deep-Learning/poutyne_Logs/classification_NW_ResNet_BS_128_FW_True_OrigLabel_True_withTestdl_1'
    infer_patch_file_name = '/home/hpcladmin/MAB/Projects/PyTorch-Deep-Learning/poutyne_Logs/classification_NW_ResNet_BS_128_FW_True_OrigLabel_True_withTestdl_1/inference_SICAPv2_Classification_prop'

    # The following file will contain the results, so its output file. This file will contain inference WSI ISUPs
    # extracted from patch level inference
    infer_slide_file_name = 'WSI_ISUP_Predictions_MultiClass_classification_prop'#'slide_classification_ISUP_prop'

    BM_File = 'slideClassification'# This file contains all original and prop WSI Labels, % GPs, and agrrement columns
    #BM_File = 'comparison_results_OrigClass_OrigMask.csv'
    # This file contains WSI level primary, secondary and ISUP Labels of original excel file and extracted labels from masks
    BM_File_path = os.path.join(project.output_dir,BM_File)
    #print(f'BM_File_path = {BM_File_path}')

    if not os.path.exists(inference_dir_path):
        # Create a new directory because it does not exist
        os.makedirs(inference_dir_path)
        print(f'Made a new dir for inferences = {inference_dir_path}')

    #*********  Inferences From trained model *************** #
    #print(f'inference_dir_path = {inference_dir_path}')
    # Assign WSI labels to classification inference of patches and gt is BM_File_path

    KNNClassifier_param_dict={'featureType':'ISUP', 'n_neighbors':7, 'L_normalization':1,
                              'weights':'distance',  # 'uniform', 'distance'
                              'algorithm':'auto',
                              'metric':'cosine', # 'minkowski', 'cityblock', 'cosine',
                              'p':1.0,
                              'reducedBow':0,# Include Nc in BOW or not, 1: do not include NC
                              }

    WSI_obj = WSI_DS(pickleListOfLists_file_name=file_name_mask,
                     featureType='ISUP',
                     useKNNClassifier = 1,# 0: do not use KNN, 1: use only KNN, 2: use both
                     #reducedBow=1, # # Include Nc in BOW or not, 1: do not include NC
                     KNNClassifier_param_dict= KNNClassifier_param_dict,
                     augmentDataset=0,
                     WSI_labels=BM_File_path,
                     inference_dir_path=inference_dir_path,
                     newSegmentationInference=False,
                     newClassificationInference=True,
                     infer_Classi_patch_file_name=infer_patch_file_name, # source file
                     infer_Classi_slide_file_name = infer_slide_file_name # destination file, output file
                     ) # Create WSI object that will contain all patch objects

    # Classify all slides
    WSI_obj.classify_slides()

    print(f'length of df list = {len(WSI_obj.df_list)}')
    WSI_obj.makeConfusionMatrix(WSI_obj.gt_BM, WSI_obj.predicted_classes, WSI_obj.classes)
    WSI_obj.makeConfusionMatrix(WSI_obj.gt_NL, WSI_obj.predicted_classes, WSI_obj.classes)

    '''
    # ****** Ground Truth ********* #
    # Extract patch GP3, GP4, GP5 from original masks for creating patch level and WSI level ground Truth
    # Now this labeling is based on Epstein criteria
    # Read the original slide classification Excel file into a DataFrame
    BM_File = 'wsi_labels.xlsx'  # Original Dataste Classification info
    datafolder = project.data_dir
    # Load the df object using original excel classification file of  DS
    BM_File_path = os.path.join(datafolder, f"{BM_File}")
    WSI_obj = WSI_DS(pickleListOfLists_file_name=file_name_mask,
                     inference_dir_path=None,#Dir of dataset masks, not needed because info is present in pickleListOfLists
                     newSegmentationInference=False,
                     newClassificationInference=False,
                     WSI_labels=BM_File_path,  # In case of labeling it is wsi_labels.xlsx (original WSI Labels)
                     infer_Classi_patch_file_name= 'patch_slideClassification',#patch labels extracted from Dataset Masks, output file
                     infer_Classi_slide_file_name = 'slideClassification',#WSI labels # destination file, output file
                     ) # Create WSI object that will contain all patch objects
    # Classify all slides
    WSI_obj.label_slides()
    '''

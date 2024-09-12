'''
This code first defines functions for building BoW representations,
training the KNN classifier, and
evaluating its performance using K-Fold cross-validation.
The main script loads the data, creates a kmeans object, builds BoW representations for each WSI, and
then performs K-Fold cross-validation to evaluate the KNN classifier with a specific k value.
'''

import torch
import torch.nn.functional as F
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from Project_DDP import project  # All parameters, paths are stored here
import pickle
import pandas as pd
import numpy as np
import random as rd

import time
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

class KNNClassifier():
    def __init__(self,
                 featureType='ISUP',
                 n_neighbors=2,
                 L_normalization=1,
                 weights='uniform', # 'uniform', 'distance'
                 algorithm='auto',
                 metric='cosine',# 'minkowski', 'cityblock', 'cosine',
                 p=1.0,
                 updateDS= 0,
                 reducedBow=0,# 0: include NC
                 patch_ISUP_features=None,
                 patch_GS_features=None,
                 labels_ISUP=None,
                 labels_GS=None,
                 BOW_df=None,
                 # If we want to add during prediction process
                 new_ISUP_features= None,
                 new_GS_features=None,
                 new_labels_ISUP=None,
                 new_labels_GS=None,
                 ):
        '''

        '''
        self.featureType = featureType
        self.n_neighbors = n_neighbors
        self.L_normalization = L_normalization
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.updateDS = updateDS
        self.reducedBow=reducedBow # if 1: NC will not be included in BOW
        self.BOW_df = BOW_df
        self.patch_ISUP_features = patch_ISUP_features
        self.patch_GS_features = patch_GS_features
        self.labels_ISUP = labels_ISUP
        self.labels_GS = labels_GS
        # If we want to add during prediction process
        self.new_ISUP_features = new_ISUP_features
        self.new_GS_features = new_GS_features
        self.new_labels_ISUP = new_labels_ISUP
        self.new_labels_GS = new_labels_GS

        # Initialize, load all features of every type and labels
        self.loadDataset()
        '''
        # Call augmentDataset() and pass values in main() portion
        aug_features_list_ISUP, aug_ISUP_label_list, aug_features_list_GS, aug_GS_label_list = self.augData()
        self.augmentDataset(new_ISUP_features=aug_features_list_ISUP,
                            new_GS_features=self.new_GS_features,
                            new_labels_ISUP=aug_ISUP_label_list,
                            new_labels_GS=self.new_labels_GS,
                            )
        '''
        self.trained_Knn_Classifier_obj = self.trainKnnClassifier()

    def findOptimalNeighbours(self, features, labels):
        # Assume you have extracted the patch features and grades for each WSI
        # patch_features is a list of feature vectors for each patch
        # grades is a list of corresponding grades for each WSI
        # [0,1,2,3,4,5] is vocabulary


        # Convert the patch features and grades to numpy arrays
        #patch_features = WSI_ISUP_features
        #grades = values_list3_coded  # GS labels

        features_np = np.array(features)
        # Normalize each row of the array
        normalized_rows_L1 = np.linalg.norm(features_np, ord=1, axis=1)
        normalized_rows_L2 = np.linalg.norm(features_np, ord=2, axis=1)
        normalized_features_L1 = features_np / normalized_rows_L1[:, np.newaxis]
        normalized_features_L2 = features_np / normalized_rows_L2[:, np.newaxis]

        grades=labels
        patch_features = np.array(features)
        # Convert the BoW representations to PyTorch tensors
        bow_representations = torch.Tensor(patch_features)
        # bow_representations_norm = bow_representations/torch.max(bow_representations)# Div by max
        # bow_representations_norm = F.normalize(bow_representations, p=2, dim=0)# L2 Norm
        bow_representations_norm = F.normalize(bow_representations, p=1, dim=0)  # L1 Norm, proved best
        # bow_representations_norm = F.normalize(bow_representations, p=0, dim=0)# L Norm,

        grades = np.array(grades)
        grades = torch.Tensor(grades)

        # Initialize a list to store the accuracy scores for each neighbour choice
        accuracy_scores_K = []

        # K neighbours
        for i, K in enumerate((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)):
            start = time.time()  # Benchmark:

            # Initialize a list to store the accuracy scores for each fold
            accuracy_scores_F = []  # All fold numbers

            # try differen folds
            for f, Fold in enumerate((2, 3, 4, 5, 6, 7, 8, 9)):
                accuracy_scores_f = []  # all folds with a given Fold f
                # print(f'f = {f}')

                # Create K-fold cross-validation iterator for every neighbourhood (K) value
                kf = KFold(n_splits=Fold, shuffle=True)

                #for train_index, test_index in kf.split(bow_representations_norm):
                for train_index, test_index in kf.split(normalized_features_L1):
                    # Split the data into training and testing sets for the current fold
                    train_data, test_data = bow_representations_norm[train_index], bow_representations_norm[test_index]
                    train_labels, test_labels = grades[train_index], grades[test_index]

                    # Create and train the KNN classifier
                    # metric='minkowski’, ‘cosine’, weights='distance'
                    knn = KNeighborsClassifier(n_neighbors=K, weights='distance', algorithm='auto', metric='cosine', p=1.0)
                    knn.fit(train_data, train_labels)

                    # Predict the grades for the test data
                    predictions = knn.predict(test_data)
                    # print(f'predictions={type(predictions)}, \ntest_labels={type(test_labels)}')

                    # Calculate the accuracy score for the fold
                    # accuracy = (predictions == test_labels).astype(float).sum() / len(test_labels)
                    # Calculate accuracy using element-wise comparison and mean
                    accuracy = np.mean(predictions == test_labels.numpy())
                    # print(f'accuracy={accuracy}')

                    accuracy = accuracy.item() * 100  # Multiply by 100 to get the accuracy as a percentage
                    accuracy_scores_f.append(accuracy)

                # Calculate the average accuracy across all folds
                average_accuracy_f = np.mean(accuracy_scores_f)
                accuracy_scores_F.append(average_accuracy_f)
                # Print the average value attained in this fold
                print(f'average_accuracy_f for {f + 2} fold cross validation and {K} neighbours : {average_accuracy_f}')
            # Print the maximum value and its index
            print(f'Best_accuracy_f for {np.argmax(accuracy_scores_F) + 2} fold_cross_validation '
                  f'and {K} neighbours : {np.amax(accuracy_scores_F)}')

            end = time.time()

            # Calculate the average accuracy across all folds
            average_accuracy_F = np.mean(accuracy_scores_F)
            accuracy_scores_K.append(average_accuracy_F)

        # Extract Best average accuracy across all neghbours K
        Best_accuracy_K = np.amax(accuracy_scores_K)  # Find the maximum value
        Best_K = np.argmax(accuracy_scores_K) + 1  # Find the maximum value's index

        # Print the maximum value and its index
        print(f'accuracy_scores_K = {accuracy_scores_K}')
        print("Maximum Accuracy:", Best_accuracy_K)
        print("Best Value for K:", Best_K)

        return Best_K

    def trainKnnClassifier(self):
        '''
        Use bow_representations_norm
        '''
        n_neighbors = self.n_neighbors
        weights = self.weights
        algorithm = self.algorithm
        metric = self.metric
        p = self.p
        L_normalization = self.L_normalization

        if self.featureType == 'ISUP':
            features = self.patch_ISUP_features
            print(f'[trainKnnClassifier()]: Shape of features ={features}')
            features_np = np.array(features)
            # Normalize the array
            normalized_array = features_np / np.linalg.norm(features_np)

            # Normalize each row of the array
            normalized_rows = np.linalg.norm(features_np, ord=L_normalization, axis=1)
            normalized_features = features_np / normalized_rows[:, np.newaxis]


            # Convert the BoW representations to PyTorch tensors
            features_T = torch.Tensor(features_np)
            features_normalized_T = F.normalize(features_T, p=L_normalization, dim=0)  # L1 Norm, proved best
            features_normalized_T_to_np = features_normalized_T.numpy()
            labels = self.labels_ISUP
            labels_np = np.array(labels)
        else:
            features = self.patch_GS_features
            features_np = np.array(features)
            # Convert the BoW representations to PyTorch tensors
            features_T = torch.Tensor(features_np)
            features_normalized_T = F.normalize(features_T, p=1, dim=0)  # L1 Norm, proved best
            labels = self.labels_GS
            labels_np = np.array(labels)

        #print(f'features_np normalized_array = {normalized_array}')
        #print(f'features_np normalized_array L1 = {normalized_features}')

        #print(f'features_np normalized_array T to np = {features_normalized_T_to_np}')
        # NOw we have found optimal value of K neighbours so train on best K and all dataset
        #print(f'Shape of features ={features_normalized_T.shape}, shape of lasbels {labels_np.shape}')
        trained_Knn_Classifier_obj = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights,
                                                          algorithm=algorithm, metric=metric, p=p)

        #print(f'\n[]:type of feature :{type(features)}, type of labels {type(labels)}')#<class 'list'>, <class 'list'>
        #print(f'\n[trainKnnClassifier]:Contents of features_normalized_T :{features_normalized_T}')
        print(f'\n[trainKnnClassifier]:Contents of labels {labels}')
        print(f'\n[trainKnnClassifier]:Length of labels list used for training {len(labels)}')

        trained_Knn_Classifier_obj.fit(normalized_features, labels_np)

        return trained_Knn_Classifier_obj

    #def KNNPredict(self,Knn_Classifier_obj, features):
    def Predict(self, features, label=None):
        '''
        features: It should be a list of lists(BOW)
        '''
        L_normalization = self.L_normalization
        updateDS = self.updateDS

        # If you want to exclude NC from the features list
        if self.reducedBow:
            for r in range(len(features)):
                #print(f'features[r] = {features[r]}')
                #print(f'features[r][] = {features[r][0]}')
                features[r]= features[r][1:]
                if sum(features[r])==0:
                    del features[r]
                    if label is not None:
                        del label[r]
                        #print(f'Label with removed label = {label}')
            #print(f'Features with Reduced BOW = {features}')

        if len(features)==0 and self.reducedBow:
            predictions= [0] # every other label's freq is zero so it must be a ISUP0 WSI
        elif len(features) > 0 :
            # Convert the new samples to numpy array
            new_samples_np = np.array(features)
            print(f'Shape new_samples ndarray = {new_samples_np.shape}')

            # Normalize each row of the array
            normalized_rows = np.linalg.norm(new_samples_np, ord=L_normalization, axis=1)
            print(f' normalized_rows = {normalized_rows}')
            normalized_new_samples_np = new_samples_np / normalized_rows[:, np.newaxis]
            print(f'normalized_new_samples_np = {normalized_new_samples_np}')

            # Convert the BoW representations of the new samples to PyTorch tensor
            new_bow_representations_T = torch.Tensor(new_samples_np)
            #print(f'new_samples new_bow_representations Tensor =\n {new_bow_representations_T}')

            new_bow_representations_T_normalized = F.normalize(new_bow_representations_T, p=1, dim=0)  # L1 Norm, proved best
            #print(f'Normalized new_bow_representations Tensor =\n {new_bow_representations_T_normalized}')
            #print(f'Normalized new_bow_representations np =\n {normalized_new_samples_np}')

            #features_shape = features.shape
            #print(f'[KNNPredict]:new_bow_representations_shape = {features_shape} and contents= {features}') # e.g. torch.Size([6])
            #print(f'[KNNPredict]:features contents= {new_bow_representations_T_normalized}') # e.g. torch.Size([6])
            #features.view(-1, 1)
            #print(f'[KNNPredict]:new_bow_representations_view = {new_bow_representations.view(-1,len(new_bow_representations))}')  # e.g. torch.Size([6])
            #predictions = self.trained_Knn_Classifier_obj.predict(features)
            #predictions = self.trained_Knn_Classifier_obj.predict(new_bow_representations_T_normalized.view(-1,6))

            predictions = self.trained_Knn_Classifier_obj.predict(normalized_new_samples_np)

            if updateDS == 1:
                for j in range(new_samples_np.shape[0]):
                    print(f'predictions ={predictions[j]} are equal to label ={label[j]}')
                    if predictions[j] == label[j]:
                        if self.featureType == 'ISUP':
                            self.augmentDataset(new_ISUP_features=[features[j]],
                                                new_labels_ISUP=[label[j]])#provide list of features even if there is one feature
                            #print(f'Augmented list in prediction : {self.patch_ISUP_features}')
                            print(f'Length of Augmented list in prediction : {len(self.patch_ISUP_features)}')
                        elif self.featureType == 'GS':
                            self.augmentDataset(new_GS_features=features,
                                                new_labels_ISUP=label)


        return predictions

    def displayClusters(self, x,cl,N):
        '''
            plt.subplot(4, 3, i + 2)  # Fancy display:
            # clg = clg.view(M, M)
            # plt.imshow(clg.cpu(), extent=(0, 1, 0, 1), origin="lower")
            plt.axis("off")
            plt.axis([0, 1, 0, 1])
            plt.tight_layout()
            plt.title("{}-NN classifier,\n t = {:.2f}s".format(K, end - start))

            #plt.show()


            plt.figure(figsize=(12, 8))
            plt.subplot(2, 3, 1)
            plt.scatter(x.cpu()[:, 0], x.cpu()[:, 1], c=cl.cpu(), s=2)
            plt.imshow(np.ones((2, 2)), extent=(0, 1, 0, 1), alpha=0)
            plt.axis("off")
            plt.axis([0, 1, 0, 1])
            plt.title("{:,} data points,\n{:,} grid points".format(N, M * M))
            '''

        # Reference sampling grid, on the unit square:
        M = 20
        tmp = torch.linspace(0, 1, M).type(dtype)
        g2, g1 = torch.meshgrid(tmp, tmp)
        g = torch.cat((g1.contiguous().view(-1, 1), g2.contiguous().view(-1, 1)), dim=1)

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 3, 1)
        plt.scatter(x.cpu()[:, 0], x.cpu()[:, 1], c=cl.cpu(), s=2)
        plt.imshow(np.ones((2, 2)), extent=(0, 1, 0, 1), alpha=0)
        plt.axis("off")
        plt.axis([0, 1, 0, 1])
        plt.title("{:,} data points,\n{:,} grid points".format(N, M * M))

    def loadDataset(self):
        # Get features from pickled file
        # Save BOWs to be used in KNN algo for WSI classification
        output_folder = project.output_dir
        file_path = os.path.join(output_folder, "BagOfWords_df.pickle")
        with open(file_path, "rb") as file:
            BagOfWords_df = pickle.load(file)
            file.close()

        self.BOW_df = BagOfWords_df
        # Write the first 5 lines
        #BagOfWords_df_head = BagOfWords_df.head(5)

        # Set the display options to show all rows and columns
        #pd.set_option('display.max_rows', None)
        #pd.set_option('display.max_columns', None)

        # Display the dataframe
        #print(self.BOW_df)

        # Print the first few lines
        #print(BagOfWords_df_head)

        # Specify the value you want to match
        #slideNo = '18B0006623J'

        # Specify the names of the selected columns as a list
        if self.reducedBow:
            selected_columns_ISUP = ['ISUP1', 'ISUP2', 'ISUP3', 'ISUP4', 'ISUP5']
            selected_columns_GS = ['GS1', 'GS2', 'GS3', 'GS4', 'GS5', 'GS6', 'GS7', 'GS8', 'GS9']
        else:
            selected_columns_ISUP = ['ISUP0', 'ISUP1', 'ISUP2', 'ISUP3', 'ISUP4', 'ISUP5']
            selected_columns_GS = ['GS0', 'GS1', 'GS2', 'GS3', 'GS4', 'GS5', 'GS6', 'GS7', 'GS8', 'GS9']

        # Filter the DataFrame based on the matching value
        #matched_rows = BagOfWords_df.loc[BagOfWords_df['Slide_No.'] == slideNo, selected_columns_ISUP]
        # Print the matched rows
        #print(matched_rows)

        file_path = os.path.join(output_folder, "BagOfWords_ISUP.pickle")
        with open(file_path, "rb") as file:
            BagOfWords_ISUP_dict = pickle.load(file)
            file.close()

        file_path = os.path.join(output_folder, "BagOfWords_GS.pickle")
        with open(file_path, "rb") as file:
            BagOfWords_GS_dict = pickle.load(file)
            file.close()
        file_path = os.path.join(output_folder, "BagOfWords_GS_Label.pickle")
        with open(file_path, "rb") as file:
            BagOfWords_GS_Label = pickle.load(file)
            file.close()
        file_path = os.path.join(output_folder, "BagOfWords_ISUP_Label.pickle")
        with open(file_path, "rb") as file:
            BagOfWords_ISUP_Label = pickle.load(file)
            file.close()

        values_list1 = list(BagOfWords_ISUP_dict.values())
        values_list2 = list(BagOfWords_GS_dict.values())
        values_list3 = list(BagOfWords_GS_Label.values())
        values_list4 = list(BagOfWords_ISUP_Label.values())
        # print(values_list1)
        # print(values_list2)
        # print(values_list3)

        print(f'[loadDataset()]:Content of ISUP labels list = {values_list4}')
        #print(f'[loadDataset()]:Content of GS labels list = {values_list3}')
        # print(f'length of labels = {len(values_list4)}')

        WSI_ISUP_features = [list(inner_dict.values()) for inner_dict in BagOfWords_ISUP_dict.values()]
        WSI_GS_features = [list(inner_dict.values()) for inner_dict in BagOfWords_GS_dict.values()]
        print(f'[loadDataset()]:WSI_ISUP_features BOW list={WSI_ISUP_features}')

        code_mapping = {'0+0': 0, '3+3': 1, '3+4': 2, '3+5': 3, '4+3': 4, '4+4': 5, '4+5': 6, '5+3': 7, '5+4': 8,
                        '5+5': 9}
        # Replace strings with codes
        values_list3_coded = [code_mapping[string] for string in values_list3]
        #print(f'[loadDataset()]:Coded GS Labels list: {values_list3_coded}')

        if self.reducedBow:
            #for r in range(len(WSI_ISUP_features)):
            for r in reversed(range(len(WSI_ISUP_features))):# to encounter index error due to dynamic reduction in list length
                # print(f'features[r] = {features[r]}')
                # print(f'features[r][] = {features[r][0]}')
                WSI_ISUP_features[r] = WSI_ISUP_features[r][1:]
                WSI_GS_features[r] = WSI_GS_features[r][1:]
                if sum(WSI_ISUP_features[r]) == 0:
                    del WSI_ISUP_features[r]
                    del WSI_GS_features[r]
                    if values_list4 is not None:
                        del values_list4[r]
                        del values_list3_coded[r]
                        #print(f'Label with removed label = {values_list4}')
            print(f'Features with Reduced BOW = {WSI_ISUP_features}')

        self.patch_ISUP_features = WSI_ISUP_features
        self.patch_GS_features = WSI_GS_features
        self.labels_ISUP = values_list4
        self.labels_GS = values_list3_coded

        print(f'[loadDataset()]:WSI_ISUP_features BOW list= {WSI_ISUP_features}')
        print(f'[loadDataset()]:Labels of WSI_ISUP_features = {self.labels_ISUP }')
        #print(f'[loadDataset()]:WSI_GS_features BOW list= {WSI_GS_features}')


    def augmentDataset(self, new_ISUP_features=None, new_GS_features=None,
                       new_labels_GS=None, new_labels_ISUP=None,
                   ):
        '''
        features: a list of list
        labels: A list of lists
        '''
        endlen = 0
        startlen = 0
        featureType=self.featureType

        # Now adjust BOW length of new features if you dont want to use frequencies of NC pathes
        if self.reducedBow==1 and featureType =='ISUP':
            # for r in range(len(WSI_ISUP_features)):
            for r in reversed(range(len(new_ISUP_features))):  # to encounter index error due to dynamic reduction in list length
                # print(f'features[r] = {features[r]}')
                # print(f'features[r][] = {features[r][0]}')
                new_ISUP_features[r] = new_ISUP_features[r][1:]
                if sum(new_ISUP_features[r]) == 0:
                    del new_ISUP_features[r]
                    if new_labels_ISUP is not None:
                        del new_labels_ISUP[r]
                        print(f'Label with removed label = {new_labels_ISUP}')
            print(f'[augmentDataset()]:new_ISUP_features with Reduced BOW = {new_ISUP_features}') #list
            print(f'[augmentDataset()]:Length of new_ISUP_features with Reduced BOW = {len(new_ISUP_features)}')
        elif self.reducedBow == 1 and featureType == 'GS':
            # for r in range(len(WSI_ISUP_features)):
            for r in reversed(range(len(new_GS_features))):  # to encounter index error due to dynamic reduction in list length
                # print(f'features[r] = {features[r]}')
                # print(f'features[r][] = {features[r][0]}')
                new_GS_features[r] = new_GS_features[r][1:]
                if sum(new_GS_features[r]) == 0:
                    del new_GS_features[r]
                    if new_labels_GS is not None:
                        del new_labels_GS[r] # it must be coded
                        print(f'Label with removed label = {new_labels_GS}')
            print(f'[augmentDataset()]:new_GS_features with Reduced BOW = {new_GS_features}')
            print(f'[augmentDataset()]:new_GS_features with Reduced BOW = {new_GS_features.shape}')

        if featureType =='ISUP':
            if new_ISUP_features is not None:
                print(f'[augmentDataset()]:new_ISUP_features = {new_ISUP_features}')
                print(f'[augmentDataset()]:new_labels_ISUP = {new_labels_ISUP}')
                features_list = self.patch_ISUP_features
                labels_list = self.labels_ISUP
                new_features = new_ISUP_features
                new_labels = new_labels_ISUP

                startlen = len(features_list)
                for i in range(len(new_features)):
                    features_list.append(new_features[i])
                    labels_list.append(new_labels[i])
                endlen = len(features_list)
                print(f'Added {endlen - startlen} features to features_list.')
            else:
                print(f'\n There are no new_ISUP_features .')
                pass
        elif featureType=='GS':
            if new_GS_features is not None:
                features_list = self.patch_GS_features
                labels_list = self.labels_GS
                new_features = new_GS_features
                new_labels = new_labels_GS

                startlen = len(features_list)
                for i in range(len(new_features)):
                    features_list.append(new_features[i])
                    labels_list.append(new_labels[i])
                endlen = len(features_list)
                print(f'Added {endlen - startlen} features to features_list.')
            else:
                print(f'\n There is no patch_GS_features List.')
        else:
            print(f'\n Provide featureType.')

        if featureType == 'ISUP':
            if (endlen - startlen) > 0:
                self.patch_ISUP_features = features_list
                self.labels_ISUP = labels_list
                #print(f'\n Augmented feature list = {self.patch_ISUP_features}')
        elif featureType == 'GS':
            if (endlen - startlen) > 0:
                self.patch_GS_features = features_list
                self.labels_GS = labels_list
        else:
            print(f'\n Please provide valid featureType.')



    def augData(self):
        '''
        Returns features and labels to be appended to original datset features(BOW)
        '''
        # **** Augment dataset ***#
        aug_features_list_ISUP = []
        aug_features_list_GS = []
        aug_ISUP_label_list = []
        aug_GS_label_list = []

        #17B00208864,"[64, 8, 4, 73, 0, 13]",0.6677,11.743,0.0008,NL:4+4,4, BM: 4+5,5, Predicted:3
        aug_features_list_ISUP.append([64, 8, 4, 73, 0, 13])
        aug_ISUP_label_list.append(4)
        aug_GS_label_list.append('4+5')

        #18B0003032A, "[59, 12, 2, 79, 0, 21]", 2.2153, 7.1656, 0.0025, NL: 4 + 3, 3, BM: 4 + 3, 3, predicted: 4
        aug_features_list_ISUP.append([59, 12, 2, 79, 0, 21])
        aug_ISUP_label_list.append(3)
        aug_GS_label_list.append('4+3')

        #17B0016566,"[102, 0, 1, 78, 0, 1]",0.7373,10.5201,0.0,4+3,3,4+3,3,4
        aug_features_list_ISUP.append([102, 0, 1, 78, 0, 1])
        aug_ISUP_label_list.append(3)
        aug_GS_label_list.append('4+3')

        #17B0018428,"[164, 8, 5, 7, 0, 0]",1.3338,0.3219,0.0,3+4,2,3+4,2,0
        aug_features_list_ISUP.append([164, 8, 5, 7, 0, 0])
        aug_ISUP_label_list.append(2)
        aug_GS_label_list.append('3+4')

        #17B0034449,"[28, 0, 0, 104, 0, 29]",0.0275,12.7812,0.5343,4+5,5,4+5,5,4
        aug_features_list_ISUP.append([0, 0, 0, 75, 0, 25])
        aug_ISUP_label_list.append(5)
        aug_GS_label_list.append('4+5')
        aug_features_list_ISUP.append([25, 0, 0, 50, 0, 25])
        aug_ISUP_label_list.append(5)
        aug_GS_label_list.append('4+5')
        aug_features_list_ISUP.append([50, 0, 0, 25, 0, 25])
        aug_ISUP_label_list.append(5)
        aug_GS_label_list.append('4+5')
        aug_features_list_ISUP.append([75, 0, 0, 15, 0, 10])
        aug_ISUP_label_list.append(5)
        aug_GS_label_list.append('4+5')
        aug_features_list_ISUP.append([90, 0, 0, 5, 0, 5])
        aug_ISUP_label_list.append(5)
        aug_GS_label_list.append('4+5')
        #18B0003896D,"[72, 0, 1, 1, 0, 12]",0.0,0.0091,4.3113,5+4,5,5+4,5,4
        aug_features_list_ISUP.append([90, 0, 1, 1, 0, 9])
        aug_ISUP_label_list.append(5)
        aug_GS_label_list.append('4+5')

        aug_features_list_ISUP.append([80, 0, 10, 5, 0, 5])
        aug_ISUP_label_list.append(5)
        aug_GS_label_list.append('4+5')

        aug_features_list_ISUP.append([70, 0, 15, 10, 0, 5])
        aug_ISUP_label_list.append(5)
        aug_GS_label_list.append('4+5')
        '''
        # to nulloify slide 18B0005124A
        #aug_features_list_ISUP.append([54, 0, 0, 22, 0, 0])
        #aug_ISUP_label_list.append(3)
        #aug_GS_label_list.append('4+3')

        aug_features_list_ISUP.append([0, 0, 55, 0, 25, 20])
        aug_ISUP_label_list.append(4)
        aug_GS_label_list.append('3+5')

        aug_features_list_ISUP.append([0, 20, 20, 20, 20, 20])
        aug_ISUP_label_list.append(4)
        aug_GS_label_list.append('4+5')
        '''
        return aug_features_list_ISUP, aug_ISUP_label_list, aug_features_list_GS, aug_GS_label_list


        # *********************************************#

    '''
    #https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
    What is the k-means algorithm?
    K-means clustering is a method for grouping n observations into K clusters. 
    It uses vector quantization and aims to assign each observation to the cluster with the nearest mean or centroid, 
    which serves as a prototype for the cluster.
    K-means clustering, a part of the unsupervised learning family in AI, is used to group similar data points together in 
    a process known as clustering. Clustering helps us understand our data in a unique way – by grouping things together 
    into – you guessed it – clusters.
    
    What is Clustering?
    Cluster analysis is a technique used in data mining and machine learning to group similar objects into clusters. 
    K-means clustering is a widely used method for cluster analysis where the aim is to partition a set of objects into 
    K clusters in such a way that the sum of the squared distances between the objects and their assigned cluster mean 
    is minimized.
    
    Hierarchical clustering and k-means clustering are two popular techniques in the field of unsupervised learning used 
    for clustering data points into distinct groups. 
    While k-means clustering divides data into a predefined number of clusters, 
    hierarchical clustering creates a hierarchical tree-like structure to represent the relationships between the clusters.
    
    Clustering is an UNSUPERVISED learning problem!
    
    First Property of K-Means Clustering Algorithm
    
    All the data points in a cluster should be similar to each other. 
        K in KNN is a parameter that refers to the number of nearest neighbors in the majority voting process.
    Second Property of K-Means Clustering Algorithm
        
    We will use KNN for SLIDES segmentation(making clusters of slides/WSI) based on GS or ISUP
    
    Understanding the Different Evaluation Metrics for Clustering
    The primary aim of clustering is not just to make clusters but to make good and meaningful ones.
    My features are patch_ISUPS or patch_GSs
    
    Understanding the Different Evaluation Metrics for Clustering
    Inertia
    Recall the first property of clusters we covered above. This is what inertia evaluates. It tells us how far the points 
    within a cluster are. So, inertia actually calculates the sum of distances of all the points within a cluster from the 
    centroid of that cluster. Normally, we use Euclidean distance as the distance metric, as long as most of the features 
    are numeric; otherwise, Manhattan distance in case most of the features are categorical.
    We calculate this for all the clusters; the final inertial value is the sum of all these distances. This distance within
    the clusters is known as intracluster distance. So, inertia gives us the sum of intracluster distances:
    
    We want the points within the same cluster to be similar to each other, right? Hence, the distance between them should 
    be as low as possible.
    
    Dunn Index
    the second property – that different clusters should be as different from each other as possible.
    This is where the Dunn index comes into action. The data points from different clusters should be as different as 
    possible.
    Along with the distance between the centroid and points, the Dunn index also takes into account the distance between 
    two clusters. This distance between the centroids of two different clusters is known as inter-cluster distance. 
    Let’s look at the formula of the Dunn index:
        Dunn index formula =min(inter cluster distance)/max(intra cluster distance)
        Dunn index is the ratio of the minimum of inter-cluster distances and maximum of intracluster distances.
    We want to maximize the Dunn index. The more the value of the Dunn index, the better the clusters will be. 
    
    Silhouette Score
    The silhouette score and plot are used to evaluate the quality of a clustering solution produced by the k-means 
    algorithm. The silhouette score measures the similarity of each point to its own cluster compared to other clusters, 
    and the silhouette plot visualizes these scores for each sample. 
    A high silhouette score indicates that the clusters are well separated, and each sample is more similar to the samples 
    in its own cluster than to samples in other clusters. 
    A silhouette score close to 0 suggests overlapping clusters, and 
    a negative score suggests poor clustering solutions.
    
    K-means clustering is a method for grouping n observations into K clusters. It uses vector quantization and aims to assign each observation to the cluster with the nearest mean or centroid, which serves as a prototype for the cluster. Originally developed for signal processing, K-means clustering is now widely used in machine learning to partition data points into K clusters based on their similarity. The goal is to minimize the sum of squared distances between the data points and their corresponding cluster centroids, resulting in clusters that are internally homogeneous and distinct from each other.
    K-means is a centroid-based algorithm or a distance-based algorithm, where we calculate the distances to assign a point to a cluster. In K-Means, each cluster is associated with a centroid.
    Optimization plays a crucial role in the k-means clustering algorithm. The goal of the optimization process is to find the best set of centroids that minimizes the sum of squared distances between each data point and its closest centroid. This process is repeated multiple times until convergence, resulting in the optimal clustering solution.
    
    How to Apply K-Means Clustering Algorithm?
    1.Choose the number of clusters k
    The first step in k-means is to pick the number of clusters, k.
    2.Select k random points from the data as centroids
    Next, we randomly select the centroid for each cluster. Let’s say we want to have 2 clusters, so k is equal to 2 here. We then randomly select the centroid:
    3.Assign all the points to the closest cluster centroid
    Once we have initialized the centroids, we assign each point to the closest cluster centroid:
    4.Recompute the centroids of newly formed clusters
    Now, once we have assigned all of the points to either cluster, the next step is to compute the centroids of newly formed clusters:
    5.  Repeat steps 3 and 4
        The step of computing the centroid and assigning all the points to the cluster based on their distance from the 
        centroid is a single iteration. But wait – when should we stop this process? It can’t run till eternity, right?
    
    Stopping Criteria for K-Means Clustering
    There are essentially three stopping criteria that can be adopted to stop the K-means algorithm:
        1.Centroids of newly formed clusters do not change
        2.Points remain in the same cluster
        3.Maximum number of iterations is reached
    We can stop the algorithm if the centroids of newly formed clusters are not changing. Even after multiple iterations, 
    if we are getting the same centroids for all the clusters, we can say that the algorithm is not learning any new 
    pattern, and it is a sign to stop the training.
    Another clear sign that we should stop the training process is if the points remain in the same cluster even after 
    training the algorithm for multiple iterations.
    Finally, we can stop the training if the maximum number of iterations is reached. Suppose we have set the number of 
    iterations as 100. The process will repeat for 100 iterations before stopping.
    
        The main objective of the K-Means algorithm is to minimize the sum of distances between the points and their respective cluster centroid.
    
        To choose the value of K, take the square root of n (sqrt(n)), where n is the total number of data points.
        Usually, an odd value of K is selected to avoid confusion between two classes of data.
    
      Pros and Cons of Using KNN
      Pros:
    
        Since the KNN algorithm requires no training before making predictions, new data can be added seamlessly,
        which will not impact the accuracy of the algorithm.
        KNN is very easy to implement. There are only two parameters required to implement KNN—the value of K and
        the distance function (e.g. Euclidean, Manhattan, etc.)
    
      Cons:
    
        The KNN algorithm does not work well with large datasets. The cost of calculating the distance between the new point
        and each existing point is huge, which degrades performance.
        Feature scaling (standardization and normalization) is required before applying the KNN algorithm to any dataset.
        Otherwise, KNN may generate wrong predictions.
    
    '''
    ################################################ Now by gpt
    '''
    # Step 1: BoW Representation
    The bag-of-words model is a way of representing text(my case: sequencew of patch images. Asequence of patch images 
    tantamounts to a WSI) data when modeling text with machine learning algorithms.
    A problem with modeling text(sequence of patches belonging to a WSI, not fix no of patches) is that it is messy, 
    and techniques like machine learning algorithms prefer well defined fixed-length inputs and outputs.

    Machine learning algorithms cannot work with raw text(seq of image) directly; the text must be converted into 
    numbers. Specifically, vectors of numbers. In language processing, the vectors x are derived from textual data, 
    in order to reflect various linguistic properties of the text.This is called feature extraction or feature encoding.
    A popular and simple method of feature extraction with text data is called the bag-of-words model of text.
    
    A bag-of-words is a representation of text that describes the occurrence of words within a document (in our case 
    a WSI). It involves two things:
        A vocabulary of known words.(my patch grades: [0,1,2,3,4,5]) or GS ['0+0','3=3', ...  ]
        A measure of the presence of known words.(may be histogram or normalized fraction of present grades)
    It is called a “bag” of words, because any information about the order or structure of words in the document 
    is discarded. The model is only concerned with whether known words occur in the document, not where in the document.

    A very common feature extraction procedures for sentences and documents is the bag-of-words approach (BOW). 
    In this approach, we look at the histogram of the words within the text, i.e. considering each word count as a feature.
'''
if __name__ == '__main__':

    #KNNClassifier_obj = KNNClassifier(WSI_ISUP_features, values_list4, BagOfWords_df)
    KNNClassifier_obj = KNNClassifier(featureType = 'ISUP', n_neighbors = 7,  L_normalization = 1,
                                      weights = 'uniform',  algorithm = 'auto', metric = 'cosine',
                                      p = 1.0,  updateDS = 0,
                                      reducedBow = 1,
                                      )

    #best_Neigbour_No = KNNClassifier_obj.findOptimalNeighbours(KNNClassifier_obj.patch_ISUP_features,
    #                                                           KNNClassifier_obj.labels_ISUP
    #                                                           )

    '''KNNClassifier_obj.augmentDataset(features_list=WSI_ISUP_features, features=aug_features_list,
                                     labels_GS_list=values_list4, labels_GS=aug_ISUP_label_list,
                                     labels_ISUP_list=None, labels_ISUP=None
                                     )
    best_Neigbour_No = KNNClassifier_obj.findOptimalNeighbours(WSI_ISUP_features, values_list4)
    print(f'best_Neigbour_No = {best_Neigbour_No}')
    trained_knn_Classifier_obj = KNNClassifier_obj.trainKnnClassifier()
    '''
    # Step 3: Predict grades for new WSI samples
    # Suppose you have new WSI samples
    new_samples_ISUP = [[96, 0, 0, 0, 0, 0], [0, 50,0,0, 0, 0], [0, 0, 50, 0, 0, 0],
                        [0, 0, 0, 50, 0, 0], [0, 0, 0, 0, 50, 0], [0, 0, 0, 0, 0, 50]
                        ]
    new_samples_GS = [[96, 0, 10, 0, 0, 0,0,0,0,0], [10, 0,10,0, 0, 0,0,0,0,0], [4, 0, 0, 40, 0, 0,0,0,0,0],
                      [40, 0, 0, 40, 0, 40,0,0,0,0], [4, 0, 40, 40, 0, 0,0,0,0,0],[4, 0, 40, 40, 0, 0,0,0,0,0]]
    labels=[0,1,2,3,4,5]
    # Convert the new samples to numpy array
    #new_samples_np = np.array(new_samples_ISUP)
    #print(f'new_samples ndarray = {new_samples_np}')

    #new_bow_representations=new_samples_np

    # Convert the BoW representations of the new samples to PyTorch tensor
    #new_bow_representations = torch.Tensor(new_bow_representations)
    #print(f'new_samples new_bow_representations Tensor =\n {new_bow_representations}')

    # Predict the grades for the new samples using the trained KNN classifier
    #predictions = KNNClassifier_obj.KNNPredict(trained_knn_Classifier_obj, new_bow_representations)

    # predict whole new list
    #prediction = KNNClassifier_obj.KNNPredict(new_samples_ISUP, L_normalization=1, label=labels,updateDS=0)
    #print("Predicted Grades:\n", prediction)
    #for i in range(len(new_bow_representations)):
    #for i in range(len(new_samples_np)):
    for i in range(len(new_samples_ISUP)):
        prediction = KNNClassifier_obj.KNNPredict([new_samples_ISUP[i]],
                                                  label=[labels[i]],
                                                  )
        # Print the predicted grades for the new samples
        print("Predicted Grades:\n", prediction)

'''The Bag-of-Words (BoW) with K-Nearest Neighbors (KNN) method:
1. Simplicity: The BoW representation and KNN classifier are both straightforward and easy to understand. The BoW 
representation provides a concise histogram-based representation of each WSI, and the KNN classifier makes predictions 
based on the similarity to its nearest neighbors.

2. Interpretability: The BoW representation allows you to interpret the importance or frequency of each feature (word) 
in the WSI. This can provide insights into which features contribute more to the WSI grade prediction.

3. Robustness to noise: The BoW representation aggregates the features in a WSI, providing a more robust representation 
that can handle variations in the number and order of patches within the WSI. Additionally, KNN is known to be robust 
to noise in the training data.

4. No training required for KNN: The KNN classifier does not require an explicit training phase since it stores the 
training samples and makes predictions based on their similarity. This can be advantageous when dealing with large or 
continuously growing datasets, as you can easily update the classifier without retraining.

However, it's important to note that the BoW with KNN method also has some limitations:

1. Loss of spatial information: The BoW representation discards the spatial information of patches within the WSI, as 
it only considers the frequency of features without considering their positions. This may limit the model's ability to 
capture fine-grained spatial patterns.

2. Vocabulary size: The size of the visual vocabulary (number of clusters in k-means) can impact the representation 
quality. A small vocabulary may not capture enough diversity in the features, while a large vocabulary may increase 
computational complexity and memory requirements.

3. Curse of dimensionality: The BoW representation can suffer from the curse of dimensionality, especially if the 
vocabulary size or the number of features is large. This can lead to increased computational costs and may require 
dimensionality reduction techniques.

4. Hyperparameter tuning: The BoW with KNN method has hyperparameters to tune, such as the number of clusters in 
k-means, the number of neighbors (k) in KNN, and the distance metric used for similarity calculation. Selecting 
appropriate values for these hyperparameters is crucial for achieving good performance.

Despite these limitations, the BoW with KNN method has been widely used and can be effective for certain image 
classification tasks, especially when interpretability and simplicity are desired.
'''
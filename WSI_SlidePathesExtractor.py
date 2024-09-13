'''
This module extract all patches related to one slide and put that in a sublist,
every sublist will contain patches of a single slide.
We will save these image and mask list and use image lists in inference by trained model to classify the slide.
Remember model will give us segmentation and we will use those tensors to fine primary and secondary grades.
'''

import os
import re
import pickle
import pandas as pd
import shutil
import utils
from Project_DDP import project  # All parameters, paths are stored here
#from Project_CPU import project  # All parameters, paths are stored here

# Paths to the image and mask folders i.e source
all_images_list = project.all_images
masks_folder = project.all_masks

print(f'[WSI_SlidePathExtractor.py]: type of all_images_list = {type(all_images_list)}')#<class 'list'>
print(f'[WSI_SlidePathExtractor.py]: type of masks_folder = {type(masks_folder)}')#<class 'list'>
print(f'[WSI_SlidePathExtractor.py]: First element of all_images_list = {masks_folder[0]}')#/home/senadmin/MAB/Datasets/SICAPv2/myMasks/16B0023614_Block_Region_3_22_3_xini_14850_yini_22437.png
################################################################

def getSlideSubListsFromImageList(filenames):
    '''
    filenames: list of file paths
    returns a list of lists. The sublists contains path of images related to a single slide.
    '''
    #print(f'type of arg filenames ={type(filenames)}') # <class 'list'>
    slide_dict = {}

    # Iterate through each filename
    for filepath in filenames:
        # Extract the filename from the file path
        filename = os.path.basename(filepath)
        #print(f'filename = {filename}')

        # Extract slide number from the file name,
        # slide_number = 18B0003896D in 18B0003896D_Block_Region_11_0_15_xini_21248_yini_38634.jpg
        # slide_number = re.match(r'^(\w+)_', filename).group(1)#16B0001851_Block_Region_1_0_0_xini_6803_yini
        slide_number = re.match(r'^([^_]+)', filename).group(1)  # 16B0001851
        #print(f'slide_number = {slide_number}, type of var slide_number = {type(slide_number)}') # <class 'str'>

        # Check if the slide number exists in the dictionary
        if slide_number in slide_dict:
            slide_dict[slide_number].append(filepath)
        else:
            slide_dict[slide_number] = [filepath]

    # Convert dictionary values to lists and return as a list of sublists
    list_of_imgLists = [sublist for sublist in slide_dict.values()]
    #print(f'length of list_of_imgLists = {len(list_of_imgLists)}')
    #print(f'first list in list_of_imgLists, len = {len(list_of_imgLists[0])}')

    return list_of_imgLists

def getMaskSubListsFromImageSubLists(ListOfLists):
    '''
    ListOfLists: A list that contains sublists. Each sublist contains file paths of patch images
    related to a single Slide.

    This function will return
    '''
    maskDirName = project.maskDirName#"mymasks"
    maskSubLists =[]

    for mylist in enumerate(ListOfLists):
        #mylist is a tuple as enumerate returns an iterator.
        #print(f"[getMaskSubListsFromImageSubLists()]: mylist length = {len(mylist[1])}") # <class 'tuple'>
        #print(f"[getMaskSubListsFromImageSubLists()]: mylist type = {type(mylist[1])}")  # <class 'list'>
        #print(f"[getMaskSubListsFromImageSubLists()]: my image list first element type = {type(mylist[1])}")  # <class 'list'>
        #print(f"[getMaskSubListsFromImageSubLists()]: my image list first sublist first element = {(mylist[1][0])}")  # <class 'list'>

        # make a list of masks with the help of a list of images.
        masks_sublist = utils.change_extension(utils.change_last_directory(mylist[1], maskDirName), ".png")
        #print(f"[]:masks_sublists list length = {len(masks_sublist)}")  # Its a List
        #print(f"[]:masks_sublists list element = {masks_sublist[0]}")  # Its a str

        #maskSubLists.extend(masks_sublist) # it gives a list with all string entries and no sublists
        maskSubLists.append(masks_sublist)

    #print(f"[getMaskSubListsFromImageSubLists()]: masks_sublists list length = {len(maskSubLists)}")  # Its a List
    #print(f"[getMaskSubListsFromImageSubLists()]: first sublists list length = {len(maskSubLists[0])}")  # Its a List
    #print(f"[getMaskSubListsFromImageSubLists()]: first element of first sublists type = {type(maskSubLists[0][0])}")  # Its a str
    #print(f"[getMaskSubListsFromImageSubLists()]: first element of first sublists = {(maskSubLists[0][0])}")  # Its a str

    return maskSubLists

def saveList(listOfLists, file_name):
    output_folder = project.output_dir

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save each sublist as a separate file with a different name
    for i, sublist in enumerate(listOfLists):
        output_filename = os.path.join(output_folder, f"{file_name}_{i}.txt")
        with open(output_filename, "w") as file:
            file.write("\n".join(sublist))



def saveList_pickle(listOfLists, file_name_WO_ext):
    output_folder = project.output_dir

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save each sublist as a separate file with a different name
    #for i, sublist in enumerate(listOfLists):
    #   output_filename = os.path.join(output_folder, f"{file_name}_{i}.pickle")
    output_filename = os.path.join(output_folder, f"{file_name_WO_ext}.pickle")
    with open(output_filename, "wb") as file:
        pickle.dump(listOfLists, file)

def loadList_pickle(file_name):
    output_folder = project.output_dir

    # Load the list object using pickle
    file_path = os.path.join(output_folder, f"{file_name}.pickle")
    with open(file_path, "rb") as file:
        list_object = pickle.load(file)

    return list_object





if __name__ == '__main__':
    '''
    # Example usage
    filenames = ["16B0001851_Block_Region_1_0_0_xini_6803_yini_59786.jpg",
                 "16B0001851_Block_Region_2_0_0_xini_1234_yini_5678.jpg",
                 "16B0001852_Block_Region_1_0_0_xini_9876_yini_5432.jpg",
                 "16B0001852_Block_Region_2_0_0_xini_1111_yini_2222.jpg",
                 "16B0001853_Block_Region_3_0_0_xini_3333_yini_4444.jpg",
                 "17B0001851_Block_Region_3_0_0_xini_3333_yini_4444.jpg"]
    sublists = split_filenames(filenames)
    for sublist in sublists:
        print(sublist)
    '''

    img_sublists = getSlideSubListsFromImageList(all_images_list)
    mask_subLists = getMaskSubListsFromImageSubLists(img_sublists)

    '''
    # Save each sublist as a separate file with a different name
    file_name = "image_slide"
    saveList(img_sublists, file_name)
    file_name = "mask_slide"
    saveList(mask_subLists, file_name)
    '''
    # Save the lists using pickle
    file_name_img = "image_slide_listOflists"
    saveList_pickle(img_sublists, file_name_img)
    file_name_mask = "mask_slide_listOflists"
    saveList_pickle(mask_subLists, file_name_mask)

    # Load the saved lists
    loaded_img_sublists = loadList_pickle(os.path.join(project.output_dir, file_name_img))
    loaded_mask_sublists = loadList_pickle(os.path.join(project.output_dir, file_name_mask))

    print("Loaded Image Sublists:")
    for sublist in loaded_img_sublists:
        print(f'length of img sublist = {len(sublist)}')
        print(f'sublist first element = {sublist[0]}')

    print("Loaded Mask Sublists:")
    for sublist in loaded_mask_sublists:
        print(f'length of mask sublist = {len(sublist)}')
        print(f'sublist first element = {sublist[0]}')


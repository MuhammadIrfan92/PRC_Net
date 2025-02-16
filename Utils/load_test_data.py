import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import cv2

import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import cv2

def load_test_image_arrays_all_channels(image_size, dir_path):
    # Constants
    SIZE_X, SIZE_Y = image_size, image_size  # Adjust these to your desired dimensions
    # channel = "RGB-CIR"  # Set your channel directory if needed

    # Capture testing image info as a list
    test_images = []

    # Directory containing the .npy files
    npy_directory = dir_path

    # Iterate through the .npy files in the directory
    for npy_path in tqdm(glob.glob(os.path.join(npy_directory, "*.npy"))):
        # Load the .npy file
        img_org = np.load(npy_path)

        # Split the 9-channel image into two 3-channel images
        img_1 = img_org[:, :, :3] # First three channels
        img_2 = img_org[:, :, 3:6] # All Channels after 3rd channel
        img_3 = img_org[:, :, 6:] # All Channels after 3rd channel

        # Convert the numpy arrays to PIL images for resizing
        img_1_pil = Image.fromarray(img_1.astype('uint8'))
        img_2_pil = Image.fromarray(img_2.astype('uint8'))
        img_3_pil = Image.fromarray(img_3.astype('uint8'))

        # Resize the images
        img_1_resized = img_1_pil.resize((SIZE_X, SIZE_Y))
        img_2_resized = img_2_pil.resize((SIZE_X, SIZE_Y))
        img_3_resized = img_3_pil.resize((SIZE_X, SIZE_Y))

        # Convert back to numpy arrays
        img_1_resized_np = np.array(img_1_resized)
        img_2_resized_np = np.array(img_2_resized)
        img_3_resized_np = np.array(img_3_resized)

        # Concatenate the resized images back into a 6-channel image
        img_resized = np.concatenate((img_1_resized_np, img_2_resized_np, img_3_resized_np), axis=-1)
        test_images.append(img_resized)
    test_images = np.array(test_images)
    return test_images

def load_test_image_arrays(image_size, dir_path):
    # Constants
    SIZE_X, SIZE_Y = image_size,image_size  # Adjust these to your desired dimensions
    # channel = "RGB-CIR"  # Set your channel directory if needed

    # Capture testing image info as a list
    test_images = []

    # Directory containing the .npy files
    npy_directory = dir_path

    # Iterate through the .npy files in the directory
    for npy_path in tqdm(glob.glob(os.path.join(npy_directory, "*.npy"))):
        # Load the .npy file
        img_org = np.load(npy_path)

        # Split the 6-channel image into two 3-channel images
        img_rgb = img_org[:, :, :3]
        img_cir = img_org[:, :, 3:]

        # Convert the numpy arrays to PIL images for resizing
        img_rgb_pil = Image.fromarray(img_rgb.astype('uint8'))
        img_cir_pil = Image.fromarray(img_cir.astype('uint8'))

        # Resize the images
        img_rgb_resized = img_rgb_pil.resize((SIZE_X, SIZE_Y))
        img_cir_resized = img_cir_pil.resize((SIZE_X, SIZE_Y))

        # Convert back to numpy arrays
        img_rgb_resized_np = np.array(img_rgb_resized)
        img_cir_resized_np = np.array(img_cir_resized)

        # Concatenate the resized images back into a 6-channel image
        img_resized = np.concatenate((img_rgb_resized_np, img_cir_resized_np), axis=2)
        test_images.append(img_resized)
    #Convert list to array for machine learning processing        
    test_images = np.array(test_images)
    return test_images


def load_test_image_arrays_sequoia(image_size, dir_path):
    # Constants
    SIZE_X, SIZE_Y = image_size,image_size  # Adjust these to your desired dimensions
    # channel = "RGB-CIR"  # Set your channel directory if needed

    # Capture testing image info as a list
    test_images = []

    # Directory containing the .npy files
    npy_directory = dir_path

    # Iterate through the .npy files in the directory
    for npy_path in tqdm(glob.glob(os.path.join(npy_directory, "*.npy"))):
        # Load the .npy file
        img_org = np.load(npy_path)

        # Split the 6-channel image into two 3-channel images
        img_rgb = img_org[:, :, :2]
        img_cir = img_org[:, :, 2:]

        # Convert the numpy arrays to PIL images for resizing
        img_rgb_pil = Image.fromarray(img_rgb.astype('uint8'))
        img_cir_pil = Image.fromarray(img_cir.astype('uint8'))

        # Resize the images
        img_rgb_resized = img_rgb_pil.resize((SIZE_X, SIZE_Y))
        img_cir_resized = img_cir_pil.resize((SIZE_X, SIZE_Y))

        # Convert back to numpy arrays
        img_rgb_resized_np = np.array(img_rgb_resized)
        img_cir_resized_np = np.array(img_cir_resized)

        # Concatenate the resized images back into a 6-channel image
        img_resized = np.concatenate((img_rgb_resized_np, img_cir_resized_np), axis=2)
        test_images.append(img_resized)
    #Convert list to array for machine learning processing        
    test_images = np.array(test_images)
    return test_images




def load_test_images(image_size, dir_path):
    #Resizing images, if needed
    SIZE_X = image_size 
    SIZE_Y = image_size
    n_classes=3 #Number of classes for segmentation

    #Capture testing image info as a list
    test_images = []

    for directory_path in glob.glob(dir_path):
        for img_path in tqdm(glob.glob(os.path.join(directory_path, "*.png"))):
            # if channel_nums == 3:
            #     img = cv2.imread(img_path)  # for RGB images     
            #     # img = cv2.imread(img_path, 0) # for Grayscale      
            #     img = cv2.resize(img, (SIZE_Y, SIZE_X))
            #     test_images.append(img)
            # else:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # for mono chanel images     
            # img = cv2.imread(img_path, 0) # for Grayscale      
            img = cv2.resize(img, (SIZE_Y, SIZE_X))
            test_images.append(img)            
        
    #Convert list to array for machine learning processing        
    test_images = np.array(test_images)
    return test_images



def load_test_masks(image_size, dir_path):
    if dir_path == "": # meaning load default 93 masks (non-blank) 003
        dir_path = f"/mnt/d/JupyterNotebooks/Experimentation/Paper-1/Datasets/2018-weedMap-dataset-release/2018-weedMap-dataset-release/Tiles/RedEdge/Testset-Multi/iMaps/Testset/"
    #Resizing images, if needed
    SIZE_X, SIZE_Y = image_size, image_size
    #Capture mask/label info as a list
    test_masks = [] 
    for directory_path in glob.glob(dir_path):
        for mask_path in tqdm(glob.glob(os.path.join(directory_path, "*.png"))):
            # mask = cv2.imread(mask_path, 0)       
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)       
            mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
            test_masks.append(mask)
            
    #Convert list to array for machine learning processing          
    test_masks = np.array(test_masks)
    return test_masks
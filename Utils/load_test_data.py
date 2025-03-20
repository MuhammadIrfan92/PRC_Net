import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import cv2

def load_test_image_arrays(image_size, dir_path):
    SIZE_X, SIZE_Y = image_size,image_size 
    test_images = []
    npy_directory = dir_path
    for npy_path in tqdm(glob.glob(os.path.join(npy_directory, "*.npy"))):
        img_org = np.load(npy_path)
        img_rgb = img_org[:, :, :3]
        img_cir = img_org[:, :, 3:]
        img_rgb_pil = Image.fromarray(img_rgb.astype('uint8'))
        img_cir_pil = Image.fromarray(img_cir.astype('uint8'))
        img_rgb_resized = img_rgb_pil.resize((SIZE_X, SIZE_Y))
        img_cir_resized = img_cir_pil.resize((SIZE_X, SIZE_Y))
        img_rgb_resized_np = np.array(img_rgb_resized)
        img_cir_resized_np = np.array(img_cir_resized)
        img_resized = np.concatenate((img_rgb_resized_np, img_cir_resized_np), axis=2)
        test_images.append(img_resized)      
    test_images = np.array(test_images)
    return test_images


def load_test_image_arrays_sequoia(image_size, dir_path):
    SIZE_X, SIZE_Y = image_size,image_size 
    test_images = []
    npy_directory = dir_path
    for npy_path in tqdm(glob.glob(os.path.join(npy_directory, "*.npy"))):
        img_org = np.load(npy_path)
        img_rgb = img_org[:, :, :2]
        img_cir = img_org[:, :, 2:]
        img_rgb_pil = Image.fromarray(img_rgb.astype('uint8'))
        img_cir_pil = Image.fromarray(img_cir.astype('uint8'))
        img_rgb_resized = img_rgb_pil.resize((SIZE_X, SIZE_Y))
        img_cir_resized = img_cir_pil.resize((SIZE_X, SIZE_Y))
        img_rgb_resized_np = np.array(img_rgb_resized)
        img_cir_resized_np = np.array(img_cir_resized)
        img_resized = np.concatenate((img_rgb_resized_np, img_cir_resized_np), axis=2)
        test_images.append(img_resized) 
    test_images = np.array(test_images)
    return test_images




def load_test_images(image_size, dir_path):
    SIZE_X = image_size 
    SIZE_Y = image_size
    n_classes=3 
    test_images = []

    for directory_path in glob.glob(dir_path):
        for img_path in tqdm(glob.glob(os.path.join(directory_path, "*.png"))):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)    
            img = cv2.resize(img, (SIZE_Y, SIZE_X))
            test_images.append(img)                  
    test_images = np.array(test_images)
    return test_images



def load_test_masks(image_size, dir_path):
    SIZE_X, SIZE_Y = image_size, image_size
    test_masks = [] 
    for directory_path in glob.glob(dir_path):
        for mask_path in tqdm(glob.glob(os.path.join(directory_path, "*.png"))):
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)       
            mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)
            test_masks.append(mask)
    test_masks = np.array(test_masks)
    return test_masks
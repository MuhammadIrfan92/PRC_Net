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

def load_train_image_arrays_all_channels(image_size, dir_path, aug_degree):
    # Constants
    SIZE_X, SIZE_Y = image_size, image_size  # Adjust these to your desired dimensions
    # channel = "1-2"  # Set your channel directory if needed

    # Capture training image info as a list
    train_images = []

    # Directory containing the .npy files
    npy_directory = dir_path

    augs = []
    if aug_degree == 1:
        augs.append('3')
    elif aug_degree == 2:
        augs.append('3')
        augs.append('6')
    elif aug_degree == 3:
        augs.append("3")
        augs.append("6")
        augs.append("7")

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
        train_images.append(img_resized)

        # Augmentations
        if aug_degree >= 1 or '1' in augs:
            # Flip horizontally
            flipped_horizontal_1 = img_1_pil.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_horizontal_2 = img_2_pil.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_horizontal_3 = img_3_pil.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_horizontal_1_resized = flipped_horizontal_1.resize((SIZE_X, SIZE_Y))
            flipped_horizontal_2_resized = flipped_horizontal_2.resize((SIZE_X, SIZE_Y))
            flipped_horizontal_3_resized = flipped_horizontal_3.resize((SIZE_X, SIZE_Y))
            flipped_horizontal_resized = np.concatenate((np.array(flipped_horizontal_1_resized), np.array(flipped_horizontal_2_resized), np.array(flipped_horizontal_3_resized)), axis=2)

        if aug_degree >= 2 or '2' in augs:
            # Flip vertically
            flipped_vertical_1 = img_1_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_2 = img_2_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_3 = img_3_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_1_resized = flipped_vertical_1.resize((SIZE_X, SIZE_Y))
            flipped_vertical_2_resized = flipped_vertical_2.resize((SIZE_X, SIZE_Y))
            flipped_vertical_3_resized = flipped_vertical_3.resize((SIZE_X, SIZE_Y))
            flipped_vertical_resized = np.concatenate((np.array(flipped_vertical_1_resized), np.array(flipped_vertical_2_resized), np.array(flipped_vertical_3_resized)), axis=2)
        
        if aug_degree >= 3 or '3' in augs:
            # Flip both horizontally and vertically
            flipped_both_1 = img_1_pil.transpose(Image.ROTATE_180)
            flipped_both_2 = img_2_pil.transpose(Image.ROTATE_180)
            flipped_both_3 = img_3_pil.transpose(Image.ROTATE_180)
            flipped_both_1_resized = flipped_both_1.resize((SIZE_X, SIZE_Y))
            flipped_both_2_resized = flipped_both_2.resize((SIZE_X, SIZE_Y))
            flipped_both_3_resized = flipped_both_3.resize((SIZE_X, SIZE_Y))
            flipped_both_resized = np.concatenate((np.array(flipped_both_1_resized), np.array(flipped_both_2_resized), np.array(flipped_both_3_resized)), axis=2)
            train_images.append(flipped_both_resized)

        if aug_degree >= 4 or '4' in augs:
            # Rotate the images by 90 degrees
            rotated_1_90 = img_1_pil.transpose(Image.ROTATE_90)
            rotated_2_90 = img_2_pil.transpose(Image.ROTATE_90)
            rotated_3_90 = img_3_pil.transpose(Image.ROTATE_90)
            rotated_1_90_resized = rotated_1_90.resize((SIZE_X, SIZE_Y))
            rotated_2_90_resized = rotated_2_90.resize((SIZE_X, SIZE_Y))
            rotated_3_90_resized = rotated_3_90.resize((SIZE_X, SIZE_Y))
            rotated_90_resized = np.concatenate((np.array(rotated_1_90_resized), np.array(rotated_2_90_resized), np.array(rotated_3_90_resized)), axis=2)

        if aug_degree >= 5 or '5' in augs:
            # Rotate the images by 270 degrees
            rotated_1_270 = img_1_pil.transpose(Image.ROTATE_270)
            rotated_2_270 = img_2_pil.transpose(Image.ROTATE_270)
            rotated_3_270 = img_3_pil.transpose(Image.ROTATE_270)
            rotated_1_270_resized = rotated_1_270.resize((SIZE_X, SIZE_Y))
            rotated_2_270_resized = rotated_2_270.resize((SIZE_X, SIZE_Y))
            rotated_3_270_resized = rotated_3_270.resize((SIZE_X, SIZE_Y))
            rotated_270_resized = np.concatenate((np.array(rotated_1_270_resized), np.array(rotated_2_270_resized), np.array(rotated_3_270_resized)), axis=2)

        if aug_degree >= 6 or '6' in augs:
            # Rotate the horizontally flipped images by 90 degrees
            rotated_flipped_1_90 = flipped_horizontal_1.transpose(Image.ROTATE_90)
            rotated_flipped_2_90 = flipped_horizontal_2.transpose(Image.ROTATE_90)
            rotated_flipped_3_90 = flipped_horizontal_3.transpose(Image.ROTATE_90)
            rotated_flipped_1_90_resized = rotated_flipped_1_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_2_90_resized = rotated_flipped_2_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_3_90_resized = rotated_flipped_3_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_90_resized = np.concatenate((np.array(rotated_flipped_1_90_resized), np.array(rotated_flipped_2_90_resized), np.array(rotated_flipped_3_90_resized)), axis=2)
            train_images.append(rotated_flipped_90_resized)

        if aug_degree >= 7 or '7' in augs:
            # Rotate the horizontally flipped images by 270 degrees
            rotated_flipped_1_270 = flipped_horizontal_1.transpose(Image.ROTATE_270)
            rotated_flipped_2_270 = flipped_horizontal_2.transpose(Image.ROTATE_270)
            rotated_flipped_3_270 = flipped_horizontal_3.transpose(Image.ROTATE_270)
            rotated_flipped_1_270_resized = rotated_flipped_1_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_2_270_resized = rotated_flipped_2_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_3_270_resized = rotated_flipped_3_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_270_resized = np.concatenate((np.array(rotated_flipped_1_270_resized), np.array(rotated_flipped_2_270_resized), np.array(rotated_flipped_3_270_resized)), axis=2)
            train_images.append(rotated_flipped_270_resized)
        else:
            pass

    # Convert list to array for machine learning processing
    train_images = np.array(train_images)
    return train_images


def load_train_image_arrays(image_size, dir_path, aug_degree):
    # Constants
    SIZE_X, SIZE_Y = image_size, image_size  # Adjust these to your desired dimensions
    # channel = "RGB-CIR"  # Set your channel directory if needed

    # Capture training image info as a list
    train_images = []

    # Directory containing the .npy files
    npy_directory = dir_path
    augs = []
    if aug_degree == 1:
        augs.append('3')
    elif aug_degree == 2:
        augs.append('3')
        augs.append('6')
    elif aug_degree == 3:
        augs.append("3")
        augs.append("6")
        augs.append("7")

    # Iterate through the .npy files in the directory
    for npy_path in tqdm(glob.glob(os.path.join(npy_directory, "*.npy"))):
        # Load the .npy file
        img_org = np.load(npy_path)

        # Split the 6-channel image into two 3-channel images
        img_rgb = img_org[:, :, :3] # First three channels
        img_cir = img_org[:, :, 3:] # All Channels after 3rd channel

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
        train_images.append(img_resized)

        # Augmentations
        if aug_degree >= 1 or '1' in augs:
            # Flip horizontally
            flipped_horizontal_rgb = img_rgb_pil.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_horizontal_cir = img_cir_pil.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_horizontal_rgb_resized = flipped_horizontal_rgb.resize((SIZE_X, SIZE_Y))
            flipped_horizontal_cir_resized = flipped_horizontal_cir.resize((SIZE_X, SIZE_Y))
            flipped_horizontal_resized = np.concatenate((np.array(flipped_horizontal_rgb_resized), np.array(flipped_horizontal_cir_resized)), axis=2)

        if aug_degree >= 2 or '2' in augs:
            # Flip vertically
            flipped_vertical_rgb = img_rgb_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_cir = img_cir_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_rgb_resized = flipped_vertical_rgb.resize((SIZE_X, SIZE_Y))
            flipped_vertical_cir_resized = flipped_vertical_cir.resize((SIZE_X, SIZE_Y))
            flipped_vertical_resized = np.concatenate((np.array(flipped_vertical_rgb_resized), np.array(flipped_vertical_cir_resized)), axis=2)
        
        if aug_degree >= 3 or '3' in augs:
            # Flip both horizontally and vertically
            flipped_both_rgb = img_rgb_pil.transpose(Image.ROTATE_180)
            flipped_both_cir = img_cir_pil.transpose(Image.ROTATE_180)
            flipped_both_rgb_resized = flipped_both_rgb.resize((SIZE_X, SIZE_Y))
            flipped_both_cir_resized = flipped_both_cir.resize((SIZE_X, SIZE_Y))
            flipped_both_resized = np.concatenate((np.array(flipped_both_rgb_resized), np.array(flipped_both_cir_resized)), axis=2)
            train_images.append(flipped_both_resized)

        if aug_degree >= 4 or '4' in augs:
            # Rotate the images by 90 degrees
            rotated_rgb_90 = img_rgb_pil.transpose(Image.ROTATE_90)
            rotated_cir_90 = img_cir_pil.transpose(Image.ROTATE_90)
            rotated_rgb_90_resized = rotated_rgb_90.resize((SIZE_X, SIZE_Y))
            rotated_cir_90_resized = rotated_cir_90.resize((SIZE_X, SIZE_Y))
            rotated_90_resized = np.concatenate((np.array(rotated_rgb_90_resized), np.array(rotated_cir_90_resized)), axis=2)

        if aug_degree >= 5 or '5' in augs:
            # Rotate the images by 270 degrees
            rotated_rgb_270 = img_rgb_pil.transpose(Image.ROTATE_270)
            rotated_cir_270 = img_cir_pil.transpose(Image.ROTATE_270)
            rotated_rgb_270_resized = rotated_rgb_270.resize((SIZE_X, SIZE_Y))
            rotated_cir_270_resized = rotated_cir_270.resize((SIZE_X, SIZE_Y))
            rotated_270_resized = np.concatenate((np.array(rotated_rgb_270_resized), np.array(rotated_cir_270_resized)), axis=2)

        if aug_degree >= 6 or '6' in augs:
            # Rotate the horizontally flipped images by 90 degrees
            rotated_flipped_rgb_90 = flipped_horizontal_rgb.transpose(Image.ROTATE_90)
            rotated_flipped_cir_90 = flipped_horizontal_cir.transpose(Image.ROTATE_90)
            rotated_flipped_rgb_90_resized = rotated_flipped_rgb_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_cir_90_resized = rotated_flipped_cir_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_90_resized = np.concatenate((np.array(rotated_flipped_rgb_90_resized), np.array(rotated_flipped_cir_90_resized)), axis=2)
            train_images.append(rotated_flipped_90_resized)

        if aug_degree >= 7 or '7' in augs:
            # Rotate the horizontally flipped images by 270 degrees
            rotated_flipped_rgb_270 = flipped_horizontal_rgb.transpose(Image.ROTATE_270)
            rotated_flipped_cir_270 = flipped_horizontal_cir.transpose(Image.ROTATE_270)
            rotated_flipped_rgb_270_resized = rotated_flipped_rgb_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_cir_270_resized = rotated_flipped_cir_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_270_resized = np.concatenate((np.array(rotated_flipped_rgb_270_resized), np.array(rotated_flipped_cir_270_resized)), axis=2)
            train_images.append(rotated_flipped_270_resized)
        else:
            pass

    # Convert list to array for machine learning processing
    train_images = np.array(train_images)
    return train_images

def load_train_image_arrays_selective_augs(image_size, dir_path, augs):
    # Constants
    SIZE_X, SIZE_Y = image_size, image_size  # Adjust these to your desired dimensions
    # channel = "RGB-CIR"  # Set your channel directory if needed

    # Capture training image info as a list
    train_images = []

    # Directory containing the .npy files
    npy_directory = dir_path

    # Iterate through the .npy files in the directory
    for npy_path in tqdm(glob.glob(os.path.join(npy_directory, "*.npy"))):
        # Load the .npy file
        img_org = np.load(npy_path)

        # Split the 6-channel image into two 3-channel images
        img_rgb = img_org[:, :, :3] # First three channels
        img_cir = img_org[:, :, 3:] # All Channels after 3rd channel

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
        train_images.append(img_resized)

        # Augmentations
        
        # Flip horizontally
        flipped_horizontal_rgb = img_rgb_pil.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_horizontal_cir = img_cir_pil.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_horizontal_rgb_resized = flipped_horizontal_rgb.resize((SIZE_X, SIZE_Y))
        flipped_horizontal_cir_resized = flipped_horizontal_cir.resize((SIZE_X, SIZE_Y))
        flipped_horizontal_resized = np.concatenate((np.array(flipped_horizontal_rgb_resized), np.array(flipped_horizontal_cir_resized)), axis=2)
        if 1 in augs:
            train_images.append(flipped_horizontal_resized)

        if 2 in augs:
            # Flip vertically
            flipped_vertical_rgb = img_rgb_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_cir = img_cir_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_rgb_resized = flipped_vertical_rgb.resize((SIZE_X, SIZE_Y))
            flipped_vertical_cir_resized = flipped_vertical_cir.resize((SIZE_X, SIZE_Y))
            flipped_vertical_resized = np.concatenate((np.array(flipped_vertical_rgb_resized), np.array(flipped_vertical_cir_resized)), axis=2)
            train_images.append(flipped_vertical_resized)
        
        if 3 in augs:
            # Flip both horizontally and vertically
            flipped_both_rgb = img_rgb_pil.transpose(Image.ROTATE_180)
            flipped_both_cir = img_cir_pil.transpose(Image.ROTATE_180)
            flipped_both_rgb_resized = flipped_both_rgb.resize((SIZE_X, SIZE_Y))
            flipped_both_cir_resized = flipped_both_cir.resize((SIZE_X, SIZE_Y))
            flipped_both_resized = np.concatenate((np.array(flipped_both_rgb_resized), np.array(flipped_both_cir_resized)), axis=2)
            train_images.append(flipped_both_resized)

        if 4 in augs:
            # Rotate the images by 90 degrees
            rotated_rgb_90 = img_rgb_pil.transpose(Image.ROTATE_90)
            rotated_cir_90 = img_cir_pil.transpose(Image.ROTATE_90)
            rotated_rgb_90_resized = rotated_rgb_90.resize((SIZE_X, SIZE_Y))
            rotated_cir_90_resized = rotated_cir_90.resize((SIZE_X, SIZE_Y))
            rotated_90_resized = np.concatenate((np.array(rotated_rgb_90_resized), np.array(rotated_cir_90_resized)), axis=2)
            train_images.append(rotated_90_resized)

        if 5 in augs:
            # Rotate the images by 270 degrees
            rotated_rgb_270 = img_rgb_pil.transpose(Image.ROTATE_270)
            rotated_cir_270 = img_cir_pil.transpose(Image.ROTATE_270)
            rotated_rgb_270_resized = rotated_rgb_270.resize((SIZE_X, SIZE_Y))
            rotated_cir_270_resized = rotated_cir_270.resize((SIZE_X, SIZE_Y))
            rotated_270_resized = np.concatenate((np.array(rotated_rgb_270_resized), np.array(rotated_cir_270_resized)), axis=2)
            train_images.append(rotated_270_resized)

        if 6 in augs:
            # Rotate the horizontally flipped images by 90 degrees
            rotated_flipped_rgb_90 = flipped_horizontal_rgb.transpose(Image.ROTATE_90)
            rotated_flipped_cir_90 = flipped_horizontal_cir.transpose(Image.ROTATE_90)
            rotated_flipped_rgb_90_resized = rotated_flipped_rgb_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_cir_90_resized = rotated_flipped_cir_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_90_resized = np.concatenate((np.array(rotated_flipped_rgb_90_resized), np.array(rotated_flipped_cir_90_resized)), axis=2)
            train_images.append(rotated_flipped_90_resized)

        if 7 in augs:
            # Rotate the horizontally flipped images by 270 degrees
            rotated_flipped_rgb_270 = flipped_horizontal_rgb.transpose(Image.ROTATE_270)
            rotated_flipped_cir_270 = flipped_horizontal_cir.transpose(Image.ROTATE_270)
            rotated_flipped_rgb_270_resized = rotated_flipped_rgb_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_cir_270_resized = rotated_flipped_cir_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_270_resized = np.concatenate((np.array(rotated_flipped_rgb_270_resized), np.array(rotated_flipped_cir_270_resized)), axis=2)
            train_images.append(rotated_flipped_270_resized)
        else:
            pass

    # Convert list to array for machine learning processing
    train_images = np.array(train_images)
    return train_images



def load_train_image_arrays_selective_augs_sequoia(image_size, dir_path, augs):
    # Constants
    SIZE_X, SIZE_Y = image_size, image_size  # Adjust these to your desired dimensions
    # channel = "RGB-CIR"  # Set your channel directory if needed

    # Capture training image info as a list
    train_images = []

    # Directory containing the .npy files
    npy_directory = dir_path

    # Iterate through the .npy files in the directory
    for npy_path in tqdm(glob.glob(os.path.join(npy_directory, "*.npy"))):
        # Load the .npy file
        img_org = np.load(npy_path)

        # Split the 6-channel image into two 3-channel images
        img_rgb = img_org[:, :, :2] # First three channels
        img_cir = img_org[:, :, 2:] # All Channels after 3rd channel

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
        train_images.append(img_resized)

        # Augmentations
        
        # Flip horizontally
        flipped_horizontal_rgb = img_rgb_pil.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_horizontal_cir = img_cir_pil.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_horizontal_rgb_resized = flipped_horizontal_rgb.resize((SIZE_X, SIZE_Y))
        flipped_horizontal_cir_resized = flipped_horizontal_cir.resize((SIZE_X, SIZE_Y))
        flipped_horizontal_resized = np.concatenate((np.array(flipped_horizontal_rgb_resized), np.array(flipped_horizontal_cir_resized)), axis=2)
        if 1 in augs:
            train_images.append(flipped_horizontal_resized)

        if 2 in augs:
            # Flip vertically
            flipped_vertical_rgb = img_rgb_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_cir = img_cir_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_rgb_resized = flipped_vertical_rgb.resize((SIZE_X, SIZE_Y))
            flipped_vertical_cir_resized = flipped_vertical_cir.resize((SIZE_X, SIZE_Y))
            flipped_vertical_resized = np.concatenate((np.array(flipped_vertical_rgb_resized), np.array(flipped_vertical_cir_resized)), axis=2)
            train_images.append(flipped_vertical_resized)
        
        if 3 in augs:
            # Flip both horizontally and vertically
            flipped_both_rgb = img_rgb_pil.transpose(Image.ROTATE_180)
            flipped_both_cir = img_cir_pil.transpose(Image.ROTATE_180)
            flipped_both_rgb_resized = flipped_both_rgb.resize((SIZE_X, SIZE_Y))
            flipped_both_cir_resized = flipped_both_cir.resize((SIZE_X, SIZE_Y))
            flipped_both_resized = np.concatenate((np.array(flipped_both_rgb_resized), np.array(flipped_both_cir_resized)), axis=2)
            train_images.append(flipped_both_resized)

        if 4 in augs:
            # Rotate the images by 90 degrees
            rotated_rgb_90 = img_rgb_pil.transpose(Image.ROTATE_90)
            rotated_cir_90 = img_cir_pil.transpose(Image.ROTATE_90)
            rotated_rgb_90_resized = rotated_rgb_90.resize((SIZE_X, SIZE_Y))
            rotated_cir_90_resized = rotated_cir_90.resize((SIZE_X, SIZE_Y))
            rotated_90_resized = np.concatenate((np.array(rotated_rgb_90_resized), np.array(rotated_cir_90_resized)), axis=2)
            train_images.append(rotated_90_resized)

        if 5 in augs:
            # Rotate the images by 270 degrees
            rotated_rgb_270 = img_rgb_pil.transpose(Image.ROTATE_270)
            rotated_cir_270 = img_cir_pil.transpose(Image.ROTATE_270)
            rotated_rgb_270_resized = rotated_rgb_270.resize((SIZE_X, SIZE_Y))
            rotated_cir_270_resized = rotated_cir_270.resize((SIZE_X, SIZE_Y))
            rotated_270_resized = np.concatenate((np.array(rotated_rgb_270_resized), np.array(rotated_cir_270_resized)), axis=2)
            train_images.append(rotated_270_resized)

        if 6 in augs:
            # Rotate the horizontally flipped images by 90 degrees
            rotated_flipped_rgb_90 = flipped_horizontal_rgb.transpose(Image.ROTATE_90)
            rotated_flipped_cir_90 = flipped_horizontal_cir.transpose(Image.ROTATE_90)
            rotated_flipped_rgb_90_resized = rotated_flipped_rgb_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_cir_90_resized = rotated_flipped_cir_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_90_resized = np.concatenate((np.array(rotated_flipped_rgb_90_resized), np.array(rotated_flipped_cir_90_resized)), axis=2)
            train_images.append(rotated_flipped_90_resized)

        if 7 in augs:
            # Rotate the horizontally flipped images by 270 degrees
            rotated_flipped_rgb_270 = flipped_horizontal_rgb.transpose(Image.ROTATE_270)
            rotated_flipped_cir_270 = flipped_horizontal_cir.transpose(Image.ROTATE_270)
            rotated_flipped_rgb_270_resized = rotated_flipped_rgb_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_cir_270_resized = rotated_flipped_cir_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_270_resized = np.concatenate((np.array(rotated_flipped_rgb_270_resized), np.array(rotated_flipped_cir_270_resized)), axis=2)
            train_images.append(rotated_flipped_270_resized)
        else:
            pass

    # Convert list to array for machine learning processing
    train_images = np.array(train_images)
    return train_images

def load_train_image_arrays_selective_augs_all_channels_sequoia(image_size, dir_path, augs):
    print('lol1')
    if augs != [1,2]:
        print('LOL')
        return "Update rest of the augs first!"
    # Constants
    print('lol2')

    SIZE_X, SIZE_Y = image_size, image_size  # Adjust these to your desired dimensions
    # channel = "RGB-CIR"  # Set your channel directory if needed

    # Capture training image info as a list
    train_images = []

    # Directory containing the .npy files
    npy_directory = dir_path

    # Iterate through the .npy files in the directory
    for npy_path in tqdm(glob.glob(os.path.join(npy_directory, "*.npy"))):
        # Load the .npy file
        img_org = np.load(npy_path)

        # Split the 6-channel image into two 3-channel images
        img_rgb = img_org[:, :, :3] # First three channels
        img_cir = img_org[:, :, 3:6] # All Channels after 3rd channel
        img_3 = img_org[:, :, 6:]

        # Convert the numpy arrays to PIL images for resizing
        img_rgb_pil = Image.fromarray(img_rgb.astype('uint8'))
        img_cir_pil = Image.fromarray(img_cir.astype('uint8'))
        img_3_pil = Image.fromarray(img_3.astype('uint8'))

        # Resize the images
        img_rgb_resized = img_rgb_pil.resize((SIZE_X, SIZE_Y))
        img_cir_resized = img_cir_pil.resize((SIZE_X, SIZE_Y))
        img_3_resized = img_3_pil.resize((SIZE_X, SIZE_Y))

        # Convert back to numpy arrays
        img_rgb_resized_np = np.array(img_rgb_resized)
        img_cir_resized_np = np.array(img_cir_resized)
        img_3_resized_np = np.array(img_3_resized)

        # Concatenate the resized images back into a 6-channel image
        img_resized = np.concatenate((img_rgb_resized_np, img_cir_resized_np, img_3_resized_np), axis=2)
        train_images.append(img_resized)

        # Augmentations
        
        # Flip horizontally
        flipped_horizontal_rgb = img_rgb_pil.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_horizontal_cir = img_cir_pil.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_horizontal_3 = img_3_pil.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_horizontal_rgb_resized = flipped_horizontal_rgb.resize((SIZE_X, SIZE_Y))
        flipped_horizontal_cir_resized = flipped_horizontal_cir.resize((SIZE_X, SIZE_Y))
        flipped_horizontal_3_resized = flipped_horizontal_3.resize((SIZE_X, SIZE_Y))
        flipped_horizontal_resized = np.concatenate((np.array(flipped_horizontal_rgb_resized), np.array(flipped_horizontal_cir_resized), np.array(flipped_horizontal_3_resized)), axis=2)
        if 1 in augs:
            train_images.append(flipped_horizontal_resized)

        if 2 in augs:
            # Flip vertically
            flipped_vertical_rgb = img_rgb_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_cir = img_cir_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_3 = img_3_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_rgb_resized = flipped_vertical_rgb.resize((SIZE_X, SIZE_Y))
            flipped_vertical_cir_resized = flipped_vertical_cir.resize((SIZE_X, SIZE_Y))
            flipped_vertical_3_resized = flipped_vertical_3.resize((SIZE_X, SIZE_Y))
            flipped_vertical_resized = np.concatenate((np.array(flipped_vertical_rgb_resized), np.array(flipped_vertical_cir_resized), np.array(flipped_vertical_3_resized)), axis=2)
            train_images.append(flipped_vertical_resized)
        
        if 3 in augs:
            # Flip both horizontally and vertically
            flipped_both_rgb = img_rgb_pil.transpose(Image.ROTATE_180)
            flipped_both_cir = img_cir_pil.transpose(Image.ROTATE_180)
            flipped_both_3 = img_3_pil.transpose(Image.ROTATE_180)
            flipped_both_rgb_resized = flipped_both_rgb.resize((SIZE_X, SIZE_Y))
            flipped_both_cir_resized = flipped_both_cir.resize((SIZE_X, SIZE_Y))
            flipped_both_3_resized = flipped_both_3.resize((SIZE_X, SIZE_Y))
            flipped_both_resized = np.concatenate((np.array(flipped_both_rgb_resized), np.array(flipped_both_cir_resized), np.array(flipped_both_3_resized)), axis=2)
            train_images.append(flipped_both_resized)

        if 4 in augs:
            # Rotate the images by 90 degrees
            rotated_rgb_90 = img_rgb_pil.transpose(Image.ROTATE_90)
            rotated_cir_90 = img_cir_pil.transpose(Image.ROTATE_90)
            rotated_rgb_90_resized = rotated_rgb_90.resize((SIZE_X, SIZE_Y))
            rotated_cir_90_resized = rotated_cir_90.resize((SIZE_X, SIZE_Y))
            rotated_90_resized = np.concatenate((np.array(rotated_rgb_90_resized), np.array(rotated_cir_90_resized)), axis=2)
            train_images.append(rotated_90_resized)

        if 5 in augs:
            # Rotate the images by 270 degrees
            rotated_rgb_270 = img_rgb_pil.transpose(Image.ROTATE_270)
            rotated_cir_270 = img_cir_pil.transpose(Image.ROTATE_270)
            rotated_rgb_270_resized = rotated_rgb_270.resize((SIZE_X, SIZE_Y))
            rotated_cir_270_resized = rotated_cir_270.resize((SIZE_X, SIZE_Y))
            rotated_270_resized = np.concatenate((np.array(rotated_rgb_270_resized), np.array(rotated_cir_270_resized)), axis=2)
            train_images.append(rotated_270_resized)

        if 6 in augs:
            # Rotate the horizontally flipped images by 90 degrees
            rotated_flipped_rgb_90 = flipped_horizontal_rgb.transpose(Image.ROTATE_90)
            rotated_flipped_cir_90 = flipped_horizontal_cir.transpose(Image.ROTATE_90)
            rotated_flipped_rgb_90_resized = rotated_flipped_rgb_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_cir_90_resized = rotated_flipped_cir_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_90_resized = np.concatenate((np.array(rotated_flipped_rgb_90_resized), np.array(rotated_flipped_cir_90_resized)), axis=2)
            train_images.append(rotated_flipped_90_resized)

        if 7 in augs:
            # Rotate the horizontally flipped images by 270 degrees
            rotated_flipped_rgb_270 = flipped_horizontal_rgb.transpose(Image.ROTATE_270)
            rotated_flipped_cir_270 = flipped_horizontal_cir.transpose(Image.ROTATE_270)
            rotated_flipped_rgb_270_resized = rotated_flipped_rgb_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_cir_270_resized = rotated_flipped_cir_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_270_resized = np.concatenate((np.array(rotated_flipped_rgb_270_resized), np.array(rotated_flipped_cir_270_resized)), axis=2)
            train_images.append(rotated_flipped_270_resized)
        else:
            pass

    # Convert list to array for machine learning processing
    train_images = np.array(train_images)
    return train_images





def load_train_images(image_size, dir_path, augs):
    #Resizing images, if needed
    SIZE_X = image_size 
    SIZE_Y = image_size
    n_classes=3 #Number of classes for segmentation

    #Capture training image info as a list
    train_images = []

    # augs = []
    # if aug_degree == 1:
    #     augs.append('3')
    # elif aug_degree == 2:
    #     augs.append('3')
    #     augs.append('6')
    # elif aug_degree == 3:
    #     augs.append("3")
    #     augs.append("6")
    #     augs.append("7")

    # Capture training image info as a list
    train_images = []    
    for directory_path in glob.glob(dir_path):
        for img_path in tqdm(glob.glob(os.path.join(directory_path, "*.png"))):
            # img_org = cv2.imread(img_path, 0) # For Grayscale
            # if channel_nums == 3: # If number of channels is three then open as multi channel
            #     print(f"Number of channels passed {channel_nums}")
            #     img_org = cv2.imread(img_path) # For RGB   
            #     img = cv2.resize(img_org, (SIZE_X, SIZE_Y))

            # else:
            img_org = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # For GrayScale/Mono-Channel   
            img = cv2.resize(img_org, (SIZE_X, SIZE_Y))
            train_images.append(img)
            flipped_horizontal = cv2.flip(img_org, 1)
            flipped_horizontal = cv2.resize(flipped_horizontal, (SIZE_X, SIZE_Y))
            if 1 in augs:
                # Flip horizontally
                train_images.append(flipped_horizontal)


            if 2 in augs:
                # Flip vertically
                flipped_vertical = cv2.flip(img_org, 0)
                flipped_vertical = cv2.resize(flipped_vertical, (SIZE_X, SIZE_Y))
                train_images.append(flipped_vertical)

            if 3 in augs:
                # Flip both horizontally and vertically
                flipped_both = cv2.flip(img_org, -1)
                flipped_both = cv2.resize(flipped_both, (SIZE_X, SIZE_Y))
                train_images.append(flipped_both)
    
            if 4 in augs:
                # Rotate the image by 90 degrees
                rotated_90 = cv2.rotate(img_org, cv2.ROTATE_90_CLOCKWISE)
                rotated_90 = cv2.resize(rotated_90, (SIZE_X, SIZE_Y))

            if 5 in augs:
                # Rotate the image by 270 degrees
                rotated_270 = cv2.rotate(img_org, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotated_270 = cv2.resize(rotated_270, (SIZE_X, SIZE_Y))

            if 6 in augs:
                # Rotate the horizontally flipped image by 90 degrees
                flipped_rotated_90 = cv2.rotate(flipped_horizontal, cv2.ROTATE_90_CLOCKWISE)
                flipped_rotated_90 = cv2.resize(flipped_rotated_90, (SIZE_X, SIZE_Y))
                train_images.append(flipped_rotated_90)

            if 7 in augs:
                # Rotate the horizontally Flliped_image by 270 degrees
                flipped_rotated_270 = cv2.rotate(flipped_horizontal, cv2.ROTATE_90_COUNTERCLOCKWISE)
                flipped_rotated_270 = cv2.resize(flipped_rotated_270, (SIZE_X, SIZE_Y))
                train_images.append(flipped_rotated_270)
                
            else:
                pass

        
    # Convert list to array for machine learning processing        
    train_images = np.array(train_images)
    return train_images


def load_train_masks(image_size, dir_path, aug_degree):
    if dir_path == "": # meaning load default 399 masks (non-blank) 000, 001, 002, 004
        dir_path = f"/mnt/c/Users/Irfan-PC/Desktop/JupyterNotebooks/Experimentation/Paper-1/Datasets/2018-weedMap-dataset-release/2018-weedMap-dataset-release/Tiles/RedEdge/Trainset-Multi/iMaps/Trainset/"
    #Resizing images, if needed
    SIZE_X = image_size 
    SIZE_Y = image_size
    n_classes=3 #Number of classes for segmentation

    augs = []
    if aug_degree == 1:
        augs.append('3')
    elif aug_degree == 2:
        augs.append('3')
        augs.append('6')
    elif aug_degree == 3:
        augs.append("3")
        augs.append("6")
        augs.append("7")

    # Capture mask/label info as a list
    train_masks = [] 
    for directory_path in glob.glob(dir_path):
        for mask_path in tqdm(glob.glob(os.path.join(directory_path, "*.png"))):
            mask_org = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)           
            mask = cv2.resize(mask_org, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)  # Otherwise ground truth changes due to interpolation
            train_masks.append(mask)

            if aug_degree >= 1 or '1' in augs:
                # Flip horizontally
                flipped_horizontal_mask = cv2.flip(mask_org, 1)
                flipped_horizontal_mask = cv2.resize(flipped_horizontal_mask, (SIZE_X, SIZE_Y))

            if aug_degree >= 2 or '2' in augs:
                # Flip vertically
                flipped_vertical_mask = cv2.flip(mask_org, 0)
                flipped_vertical_mask = cv2.resize(flipped_vertical_mask, (SIZE_X, SIZE_Y))

            if aug_degree >= 3 or '3' in augs:
                # Flip both horizontally and vertically
                flipped_both_mask = cv2.flip(mask_org, -1)
                flipped_both_mask = cv2.resize(flipped_both_mask, (SIZE_X, SIZE_Y))
                train_masks.append(flipped_both_mask) 

            if aug_degree >= 4 or '4' in augs:
                # Rotate the masks by 90 degrees
                rotated_mask_90 = cv2.rotate(mask_org, cv2.ROTATE_90_CLOCKWISE)
                rotated_mask_90 = cv2.resize(rotated_mask_90, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)

            if aug_degree >= 5 or '5' in augs:
                # Rotate the masks by 270 degrees
                rotated_mask_270 = cv2.rotate(mask_org, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotated_mask_270 = cv2.resize(rotated_mask_270, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)

            if aug_degree >= 6 or '6' in augs:
                # Rotate the horizontally flipped mask by 90 degrees
                flipped_rotated_mask_90 = cv2.rotate(flipped_horizontal_mask, cv2.ROTATE_90_CLOCKWISE)
                flipped_rotated_mask_90 = cv2.resize(flipped_rotated_mask_90, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
                train_masks.append(flipped_rotated_mask_90)

            if aug_degree >= 7 or '7' in augs:
                # Rotate the horizontally flipped mask by 270 degrees
                flipped_rotated_mask_270 = cv2.rotate(flipped_horizontal_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                flipped_rotated_mask_270 = cv2.resize(flipped_rotated_mask_270, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
                train_masks.append(flipped_rotated_mask_270)
            else:
                pass
  
    # Convert list to array for machine learning processing          
    train_masks = np.array(train_masks)
    return train_masks


def load_train_masks_selective_augs(image_size, dir_path, augs):
    if dir_path == "": # meaning load default 399 masks (non-blank) 000, 001, 002, 004
        dir_path = f"/mnt/c/Users/Irfan-PC/Desktop/JupyterNotebooks/Experimentation/Paper-1/Datasets/2018-weedMap-dataset-release/2018-weedMap-dataset-release/Tiles/RedEdge/Trainset-Multi/iMaps/Trainset/"
    #Resizing images, if needed
    SIZE_X = image_size 
    SIZE_Y = image_size
    n_classes=3 #Number of classes for segmentation

    # Capture mask/label info as a list
    train_masks = [] 
    for directory_path in glob.glob(dir_path):
        for mask_path in tqdm(glob.glob(os.path.join(directory_path, "*.png"))):
            mask_org = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)           
            mask = cv2.resize(mask_org, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)  # Otherwise ground truth changes due to interpolation
            train_masks.append(mask)

            
            # Flip horizontally
            flipped_horizontal_mask = cv2.flip(mask_org, 1)
            flipped_horizontal_mask = cv2.resize(flipped_horizontal_mask, (SIZE_X, SIZE_Y))
            if 1 in augs:
                train_masks.append(flipped_horizontal_mask)

            if 2 in augs:
                # Flip vertically
                flipped_vertical_mask = cv2.flip(mask_org, 0)
                flipped_vertical_mask = cv2.resize(flipped_vertical_mask, (SIZE_X, SIZE_Y))
                train_masks.append(flipped_vertical_mask)

            if 3 in augs:
                # Flip both horizontally and vertically
                flipped_both_mask = cv2.flip(mask_org, -1)
                flipped_both_mask = cv2.resize(flipped_both_mask, (SIZE_X, SIZE_Y))
                train_masks.append(flipped_both_mask) 

            if 4 in augs:
                # Rotate the masks by 90 degrees
                rotated_mask_90 = cv2.rotate(mask_org, cv2.ROTATE_90_CLOCKWISE)
                rotated_mask_90 = cv2.resize(rotated_mask_90, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
                train_masks.append(rotated_mask_90)

            if 5 in augs:
                # Rotate the masks by 270 degrees
                rotated_mask_270 = cv2.rotate(mask_org, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotated_mask_270 = cv2.resize(rotated_mask_270, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
                train_masks.append(rotated_mask_270)

            if 6 in augs:
                # Rotate the horizontally flipped mask by 90 degrees
                flipped_rotated_mask_90 = cv2.rotate(flipped_horizontal_mask, cv2.ROTATE_90_CLOCKWISE)
                flipped_rotated_mask_90 = cv2.resize(flipped_rotated_mask_90, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
                train_masks.append(flipped_rotated_mask_90)

            if 7 in augs:
                # Rotate the horizontally flipped mask by 270 degrees
                flipped_rotated_mask_270 = cv2.rotate(flipped_horizontal_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                flipped_rotated_mask_270 = cv2.resize(flipped_rotated_mask_270, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
                train_masks.append(flipped_rotated_mask_270)
            else:
                pass
  
    # Convert list to array for machine learning processing          
    train_masks = np.array(train_masks)
    return train_masks
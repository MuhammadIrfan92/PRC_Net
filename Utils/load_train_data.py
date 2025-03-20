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


def load_train_image_arrays_selective_augs(image_size, dir_path, augs):
    SIZE_X, SIZE_Y = image_size, image_size
    train_images = []
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
        train_images.append(img_resized)
        flipped_horizontal_rgb = img_rgb_pil.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_horizontal_cir = img_cir_pil.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_horizontal_rgb_resized = flipped_horizontal_rgb.resize((SIZE_X, SIZE_Y))
        flipped_horizontal_cir_resized = flipped_horizontal_cir.resize((SIZE_X, SIZE_Y))
        flipped_horizontal_resized = np.concatenate((np.array(flipped_horizontal_rgb_resized), np.array(flipped_horizontal_cir_resized)), axis=2)
        if 1 in augs:
            train_images.append(flipped_horizontal_resized)

        if 2 in augs:
            flipped_vertical_rgb = img_rgb_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_cir = img_cir_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_rgb_resized = flipped_vertical_rgb.resize((SIZE_X, SIZE_Y))
            flipped_vertical_cir_resized = flipped_vertical_cir.resize((SIZE_X, SIZE_Y))
            flipped_vertical_resized = np.concatenate((np.array(flipped_vertical_rgb_resized), np.array(flipped_vertical_cir_resized)), axis=2)
            train_images.append(flipped_vertical_resized)
        
        if 3 in augs:
            flipped_both_rgb = img_rgb_pil.transpose(Image.ROTATE_180)
            flipped_both_cir = img_cir_pil.transpose(Image.ROTATE_180)
            flipped_both_rgb_resized = flipped_both_rgb.resize((SIZE_X, SIZE_Y))
            flipped_both_cir_resized = flipped_both_cir.resize((SIZE_X, SIZE_Y))
            flipped_both_resized = np.concatenate((np.array(flipped_both_rgb_resized), np.array(flipped_both_cir_resized)), axis=2)
            train_images.append(flipped_both_resized)

        if 4 in augs:
            rotated_rgb_90 = img_rgb_pil.transpose(Image.ROTATE_90)
            rotated_cir_90 = img_cir_pil.transpose(Image.ROTATE_90)
            rotated_rgb_90_resized = rotated_rgb_90.resize((SIZE_X, SIZE_Y))
            rotated_cir_90_resized = rotated_cir_90.resize((SIZE_X, SIZE_Y))
            rotated_90_resized = np.concatenate((np.array(rotated_rgb_90_resized), np.array(rotated_cir_90_resized)), axis=2)
            train_images.append(rotated_90_resized)

        if 5 in augs:
            rotated_rgb_270 = img_rgb_pil.transpose(Image.ROTATE_270)
            rotated_cir_270 = img_cir_pil.transpose(Image.ROTATE_270)
            rotated_rgb_270_resized = rotated_rgb_270.resize((SIZE_X, SIZE_Y))
            rotated_cir_270_resized = rotated_cir_270.resize((SIZE_X, SIZE_Y))
            rotated_270_resized = np.concatenate((np.array(rotated_rgb_270_resized), np.array(rotated_cir_270_resized)), axis=2)
            train_images.append(rotated_270_resized)

        if 6 in augs:
            rotated_flipped_rgb_90 = flipped_horizontal_rgb.transpose(Image.ROTATE_90)
            rotated_flipped_cir_90 = flipped_horizontal_cir.transpose(Image.ROTATE_90)
            rotated_flipped_rgb_90_resized = rotated_flipped_rgb_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_cir_90_resized = rotated_flipped_cir_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_90_resized = np.concatenate((np.array(rotated_flipped_rgb_90_resized), np.array(rotated_flipped_cir_90_resized)), axis=2)
            train_images.append(rotated_flipped_90_resized)

        if 7 in augs:
            rotated_flipped_rgb_270 = flipped_horizontal_rgb.transpose(Image.ROTATE_270)
            rotated_flipped_cir_270 = flipped_horizontal_cir.transpose(Image.ROTATE_270)
            rotated_flipped_rgb_270_resized = rotated_flipped_rgb_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_cir_270_resized = rotated_flipped_cir_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_270_resized = np.concatenate((np.array(rotated_flipped_rgb_270_resized), np.array(rotated_flipped_cir_270_resized)), axis=2)
            train_images.append(rotated_flipped_270_resized)
        else:
            pass

    train_images = np.array(train_images)
    return train_images



def load_train_image_arrays_selective_augs_sequoia(image_size, dir_path, augs):
    SIZE_X, SIZE_Y = image_size, image_size 
    train_images = []
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
        train_images.append(img_resized)
        flipped_horizontal_rgb = img_rgb_pil.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_horizontal_cir = img_cir_pil.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_horizontal_rgb_resized = flipped_horizontal_rgb.resize((SIZE_X, SIZE_Y))
        flipped_horizontal_cir_resized = flipped_horizontal_cir.resize((SIZE_X, SIZE_Y))
        flipped_horizontal_resized = np.concatenate((np.array(flipped_horizontal_rgb_resized), np.array(flipped_horizontal_cir_resized)), axis=2)
        if 1 in augs:
            train_images.append(flipped_horizontal_resized)

        if 2 in augs:
            flipped_vertical_rgb = img_rgb_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_cir = img_cir_pil.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_vertical_rgb_resized = flipped_vertical_rgb.resize((SIZE_X, SIZE_Y))
            flipped_vertical_cir_resized = flipped_vertical_cir.resize((SIZE_X, SIZE_Y))
            flipped_vertical_resized = np.concatenate((np.array(flipped_vertical_rgb_resized), np.array(flipped_vertical_cir_resized)), axis=2)
            train_images.append(flipped_vertical_resized)
        
        if 3 in augs:
            flipped_both_rgb = img_rgb_pil.transpose(Image.ROTATE_180)
            flipped_both_cir = img_cir_pil.transpose(Image.ROTATE_180)
            flipped_both_rgb_resized = flipped_both_rgb.resize((SIZE_X, SIZE_Y))
            flipped_both_cir_resized = flipped_both_cir.resize((SIZE_X, SIZE_Y))
            flipped_both_resized = np.concatenate((np.array(flipped_both_rgb_resized), np.array(flipped_both_cir_resized)), axis=2)
            train_images.append(flipped_both_resized)

        if 4 in augs:
            rotated_rgb_90 = img_rgb_pil.transpose(Image.ROTATE_90)
            rotated_cir_90 = img_cir_pil.transpose(Image.ROTATE_90)
            rotated_rgb_90_resized = rotated_rgb_90.resize((SIZE_X, SIZE_Y))
            rotated_cir_90_resized = rotated_cir_90.resize((SIZE_X, SIZE_Y))
            rotated_90_resized = np.concatenate((np.array(rotated_rgb_90_resized), np.array(rotated_cir_90_resized)), axis=2)
            train_images.append(rotated_90_resized)

        if 5 in augs:
            rotated_rgb_270 = img_rgb_pil.transpose(Image.ROTATE_270)
            rotated_cir_270 = img_cir_pil.transpose(Image.ROTATE_270)
            rotated_rgb_270_resized = rotated_rgb_270.resize((SIZE_X, SIZE_Y))
            rotated_cir_270_resized = rotated_cir_270.resize((SIZE_X, SIZE_Y))
            rotated_270_resized = np.concatenate((np.array(rotated_rgb_270_resized), np.array(rotated_cir_270_resized)), axis=2)
            train_images.append(rotated_270_resized)

        if 6 in augs:
            rotated_flipped_rgb_90 = flipped_horizontal_rgb.transpose(Image.ROTATE_90)
            rotated_flipped_cir_90 = flipped_horizontal_cir.transpose(Image.ROTATE_90)
            rotated_flipped_rgb_90_resized = rotated_flipped_rgb_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_cir_90_resized = rotated_flipped_cir_90.resize((SIZE_X, SIZE_Y))
            rotated_flipped_90_resized = np.concatenate((np.array(rotated_flipped_rgb_90_resized), np.array(rotated_flipped_cir_90_resized)), axis=2)
            train_images.append(rotated_flipped_90_resized)

        if 7 in augs:
            rotated_flipped_rgb_270 = flipped_horizontal_rgb.transpose(Image.ROTATE_270)
            rotated_flipped_cir_270 = flipped_horizontal_cir.transpose(Image.ROTATE_270)
            rotated_flipped_rgb_270_resized = rotated_flipped_rgb_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_cir_270_resized = rotated_flipped_cir_270.resize((SIZE_X, SIZE_Y))
            rotated_flipped_270_resized = np.concatenate((np.array(rotated_flipped_rgb_270_resized), np.array(rotated_flipped_cir_270_resized)), axis=2)
            train_images.append(rotated_flipped_270_resized)
        else:
            pass

    train_images = np.array(train_images)
    return train_images




def load_train_masks_selective_augs(image_size, dir_path, augs):
    SIZE_X = image_size 
    SIZE_Y = image_size
    n_classes=3
    train_masks = [] 
    for directory_path in glob.glob(dir_path):
        for mask_path in tqdm(glob.glob(os.path.join(directory_path, "*.png"))):
            mask_org = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)           
            mask = cv2.resize(mask_org, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
            train_masks.append(mask)
            flipped_horizontal_mask = cv2.flip(mask_org, 1)
            flipped_horizontal_mask = cv2.resize(flipped_horizontal_mask, (SIZE_X, SIZE_Y))
            if 1 in augs:
                train_masks.append(flipped_horizontal_mask)

            if 2 in augs:
                flipped_vertical_mask = cv2.flip(mask_org, 0)
                flipped_vertical_mask = cv2.resize(flipped_vertical_mask, (SIZE_X, SIZE_Y))
                train_masks.append(flipped_vertical_mask)

            if 3 in augs:
                flipped_both_mask = cv2.flip(mask_org, -1)
                flipped_both_mask = cv2.resize(flipped_both_mask, (SIZE_X, SIZE_Y))
                train_masks.append(flipped_both_mask) 

            if 4 in augs:
                rotated_mask_90 = cv2.rotate(mask_org, cv2.ROTATE_90_CLOCKWISE)
                rotated_mask_90 = cv2.resize(rotated_mask_90, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
                train_masks.append(rotated_mask_90)

            if 5 in augs:
                rotated_mask_270 = cv2.rotate(mask_org, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotated_mask_270 = cv2.resize(rotated_mask_270, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
                train_masks.append(rotated_mask_270)

            if 6 in augs:
                flipped_rotated_mask_90 = cv2.rotate(flipped_horizontal_mask, cv2.ROTATE_90_CLOCKWISE)
                flipped_rotated_mask_90 = cv2.resize(flipped_rotated_mask_90, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
                train_masks.append(flipped_rotated_mask_90)

            if 7 in augs:
                flipped_rotated_mask_270 = cv2.rotate(flipped_horizontal_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                flipped_rotated_mask_270 = cv2.resize(flipped_rotated_mask_270, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
                train_masks.append(flipped_rotated_mask_270)
            else:
                pass
          
    train_masks = np.array(train_masks)
    return train_masks
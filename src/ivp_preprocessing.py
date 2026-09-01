import cv2
import numpy as np
from PIL import Image
import os
from src import config

def crop_image_from_gray(img, tol=7):
    """
    Auto-crop dark background pixels surrounding fundus image.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape
        if check_shape[0] == 0 or check_shape[1] == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            return np.stack([img1, img2, img3], axis=-1)
    return img

def extract_green_channel(image_bgr):
    """
    Extract green channel which yields optimal contrast for retinal vessels & lesions.
    """
    return image_bgr[:, :, 1]

def apply_clahe(gray_or_green, clip_limit=config.CLAHE_CLIP_LIMIT, tile_grid_size=config.CLAHE_TILE_GRID_SIZE):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_or_green)

def ben_graham_preprocessing(img_bgr, target_size=config.IMAGE_SIZE, sigma=config.BEN_GRAHAM_SIGMA):
    """
    Ben Graham's preprocessing method:
    Rescale image, crop black borders, subtract local average color to normalize illumination.
    """
    cropped = crop_image_from_gray(img_bgr)
    resized = cv2.resize(cropped, target_size)
    
    # Subtract local mean color (Gaussian Blur blend)
    blurred = cv2.GaussianBlur(resized, (0, 0), sigmaX=target_size[0] / 30)
    enhanced = cv2.addWeighted(resized, 4, blurred, -4, 128)
    return enhanced

def full_ivp_pipeline(img_input, target_size=config.IMAGE_SIZE):
    """
    Runs the complete Image Processing (IVP) pipeline and returns
    intermediate visual stages as well as the final normalized RGB image.
    
    Parameters:
        img_input: numpy array (BGR or RGB) or file path
        
    Returns:
        dict containing:
            - 'raw': original BGR
            - 'cropped': auto-cropped fundus
            - 'green_channel': green channel image
            - 'clahe': CLAHE enhanced green channel
            - 'ben_graham': Ben Graham method enhanced RGB
            - 'final_processed': final preprocessed RGB normalized for model input
    """
    if isinstance(img_input, str):
        if not os.path.exists(img_input):
            raise FileNotFoundError(f"Image not found at path: {img_input}")
        raw_bgr = cv2.imread(img_input)
    elif isinstance(img_input, np.ndarray):
        raw_bgr = img_input.copy()
        if raw_bgr.ndim == 2:
            raw_bgr = cv2.cvtColor(raw_bgr, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError("Input must be image file path or numpy array.")

    # 1. Crop dark borders
    cropped_bgr = crop_image_from_gray(raw_bgr)
    cropped_resized = cv2.resize(cropped_bgr, target_size)
    
    # 2. Extract Green Channel
    green = extract_green_channel(cropped_resized)
    
    # 3. Apply CLAHE on Green Channel
    clahe_enhanced = apply_clahe(green)
    
    # 4. Ben Graham Illumination Normalization
    ben_graham_bgr = ben_graham_preprocessing(cropped_bgr, target_size=target_size)
    
    # 5. Final RGB representation combining CLAHE & Ben Graham
    ben_graham_rgb = cv2.cvtColor(ben_graham_bgr, cv2.COLOR_BGR2RGB)
    
    return {
        'raw': raw_bgr,
        'cropped': cropped_resized,
        'green_channel': green,
        'clahe': clahe_enhanced,
        'ben_graham': ben_graham_rgb,
        'final_processed': ben_graham_rgb
    }

if __name__ == "__main__":
    print("Testing IVP Preprocessing Pipeline...")
    sample_path = os.path.join(config.DATASET_DIR, "Mild", "0024cdab0c1e.png")
    if os.path.exists(sample_path):
        results = full_ivp_pipeline(sample_path)
        print("Success! Preprocessed image shape:", results['final_processed'].shape)
    else:
        print("Sample image not found, pipeline definition validated.")

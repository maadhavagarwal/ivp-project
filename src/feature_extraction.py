import cv2
import numpy as np
from src.ivp_preprocessing import extract_green_channel, apply_clahe, crop_image_from_gray

def extract_retina_circle_mask(green_channel):
    """
    Extract binary mask of valid circular retinal region excluding dark outer background.
    """
    _, mask = cv2.threshold(green_channel, 12, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def extract_optic_disc_mask(green_clahe, retina_mask):
    """
    Locate and mask the Optic Disc (brightest circular region in retinal fundus).
    """
    masked_green = cv2.bitwise_and(green_clahe, green_clahe, mask=retina_mask)
    
    # Threshold top 2.5% brightest pixels in retina
    valid_pixels = masked_green[retina_mask > 0]
    if len(valid_pixels) == 0:
        return np.zeros_like(green_clahe)
        
    threshold_val = np.percentile(valid_pixels, 97.5)
    _, bright_mask = cv2.threshold(masked_green, int(threshold_val), 255, cv2.THRESH_BINARY)
    
    # Morphological opening to isolate Optic Disc candidate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    disc_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    
    # Find largest contour (Optic Disc)
    contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_od_mask = np.zeros_like(green_clahe)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_od_mask, [largest_cnt], -1, 255, -1)
        # Dilate OD mask slightly to cover margin
        final_od_mask = cv2.dilate(final_od_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        
    return final_od_mask

def segment_blood_vessels(green_channel, retina_mask):
    """
    Extract vessel tree structure using Morphological Black-Hat & Adaptive Thresholding.
    """
    clahe_green = apply_clahe(green_channel, clip_limit=3.0)
    
    # Morphological Black-Hat transform to isolate dark vessel structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(clahe_green, cv2.MORPH_BLACKHAT, kernel)
    
    # Adaptive thresholding to segment vessels
    vessels_raw = cv2.adaptiveThreshold(blackhat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 13, -3)
    
    # Mask to retina region
    vessels = cv2.bitwise_and(vessels_raw, vessels_raw, mask=retina_mask)
    vessels = cv2.medianBlur(vessels, 3)
    return vessels

def detect_exudates(green_clahe, od_mask, retina_mask):
    """
    Detect hard exudates (bright lipid deposits) excluding Optic Disc & non-retina background.
    """
    # Exclude Optic Disc and non-retina background
    search_mask = cv2.bitwise_and(retina_mask, cv2.bitwise_not(od_mask))
    masked_green = cv2.bitwise_and(green_clahe, green_clahe, mask=search_mask)
    
    valid_pixels = masked_green[search_mask > 0]
    if len(valid_pixels) == 0:
        return np.zeros_like(green_clahe)

    # Exudates are bright yellow/white lesions (top 1.5% intensity outside OD)
    thresh_val = max(int(np.percentile(valid_pixels, 98.5)), 200)
    _, exudates_raw = cv2.threshold(masked_green, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Remove tiny noise with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    exudates = cv2.morphologyEx(exudates_raw, cv2.MORPH_OPEN, kernel)
    return exudates

def detect_red_lesions(green_clahe, vessel_mask, retina_mask):
    """
    Detect microaneurysms and hemorrhages (small dark red spots) using morphological connected components.
    Filters out background noise and vessel trunks.
    """
    # Exclude vessels and non-retina background
    search_mask = cv2.bitwise_and(retina_mask, cv2.bitwise_not(vessel_mask))
    masked_green = cv2.bitwise_and(green_clahe, green_clahe, mask=search_mask)
    
    # Bottom-Hat filter to extract isolated dark spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bottom_hat = cv2.morphologyEx(masked_green, cv2.MORPH_BLACKHAT, kernel)
    
    # High contrast threshold for true microaneurysms
    _, lesions_raw = cv2.threshold(bottom_hat, 32, 255, cv2.THRESH_BINARY)
    lesions_raw = cv2.bitwise_and(lesions_raw, lesions_raw, mask=search_mask)
    
    # Morphological connected component analysis to filter noise
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lesions_raw)
    clean_lesions = np.zeros_like(green_clahe)
    microaneurysm_count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Filter microaneurysms by area: 3 px to 80 px in 224x224 image
        if 3 <= area <= 80:
            clean_lesions[labels == i] = 255
            microaneurysm_count += 1
            
    return clean_lesions, microaneurysm_count

def extract_all_ivp_features(img_bgr):
    """
    Extract accurate quantitative feature vector & biomarker measurements based on IVP.
    
    Returns:
        tuple: (feature_dict, visualization_masks_dict)
    """
    cropped = crop_image_from_gray(img_bgr)
    resized = cv2.resize(cropped, (224, 224))
    green = extract_green_channel(resized)
    green_clahe = apply_clahe(green)
    
    # 1. Retina Circle Mask
    retina_mask = extract_retina_circle_mask(green)
    retina_pixel_count = max(np.sum(retina_mask > 0), 1)
    
    # 2. Optic Disc Mask
    od_mask = extract_optic_disc_mask(green_clahe, retina_mask)
    
    # 3. Vessels Segmentation
    vessels = segment_blood_vessels(green, retina_mask)
    
    # 4. Hard Exudates Detection
    exudates = detect_exudates(green_clahe, od_mask, retina_mask)
    
    # 5. Red Lesions (Microaneurysms) Detection
    red_lesions, lesion_count = detect_red_lesions(green_clahe, vessels, retina_mask)
    
    # Compute precise quantitative ratios
    vessel_density = float(np.sum(vessels > 0)) / retina_pixel_count
    exudate_ratio = float(np.sum(exudates > 0)) / retina_pixel_count
    lesion_ratio = float(np.sum(red_lesions > 0)) / retina_pixel_count
    
    exudate_cnts, _ = cv2.findContours(exudates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = {
        'vessel_density': vessel_density,
        'exudate_ratio': exudate_ratio,
        'exudate_count': len(exudate_cnts),
        'lesion_ratio': lesion_ratio,
        'lesion_count': lesion_count,
        'mean_green': float(np.mean(green[retina_mask > 0])) if np.sum(retina_mask > 0) > 0 else 0.0,
        'std_green': float(np.std(green[retina_mask > 0])) if np.sum(retina_mask > 0) > 0 else 0.0,
        'clahe_contrast': float(np.std(green_clahe[retina_mask > 0])) if np.sum(retina_mask > 0) > 0 else 0.0
    }
    
    masks = {
        'optic_disc': od_mask,
        'vessels': vessels,
        'exudates': exudates,
        'red_lesions': red_lesions
    }
    
    return features, masks

def get_biomarker_clinical_grade(features):
    """
    Computes rule-based clinical DR grade based on extracted IVP biomarkers.
    Follows International Clinical Diabetic Retinopathy Disease Severity Scale.
    """
    lesion_cnt = features['lesion_count']
    exudate_ratio = features['exudate_ratio']
    
    if lesion_cnt <= 2 and exudate_ratio < 0.001:
        return 0, "No DR"
    elif lesion_cnt <= 12 and exudate_ratio < 0.005:
        return 1, "Mild DR"
    elif lesion_cnt <= 30 or (0.005 <= exudate_ratio < 0.02):
        return 2, "Moderate DR"
    elif lesion_cnt <= 55 or (0.02 <= exudate_ratio < 0.04):
        return 3, "Severe DR"
    else:
        return 4, "Proliferative DR"

if __name__ == "__main__":
    import os
    from src import config
    print("Testing Refined Classical IVP Feature Extraction...")
    sample_path = os.path.join(config.DATASET_DIR, "Mild", "0024cdab0c1e.png")
    if os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        feats, masks = extract_all_ivp_features(img)
        b_grade, b_name = get_biomarker_clinical_grade(feats)
        print("Refined Extracted Biomarkers:", feats)
        print(f"Biomarker Clinical Grade: Grade {b_grade} ({b_name})")

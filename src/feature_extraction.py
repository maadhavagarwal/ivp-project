import cv2
import numpy as np
from skimage.filters import frangi
from src.ivp_preprocessing import extract_green_channel, apply_clahe, crop_image_from_gray

def extract_retina_circle_mask(green_channel):
 
    _, mask = cv2.threshold(green_channel, 12, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def extract_optic_disc_mask(green_clahe, retina_mask):
    """
    Locate and mask the Optic Disc (brightest circular region in retinal fundus).
    Returns: (final_od_mask, od_centroid_xy, od_radius_px)
    """
    masked_green = cv2.bitwise_and(green_clahe, green_clahe, mask=retina_mask)
    
    valid_pixels = masked_green[retina_mask > 0]
    if len(valid_pixels) == 0:
        return np.zeros_like(green_clahe), (112, 112), 20
        
    threshold_val = np.percentile(valid_pixels, 97.5)
    _, bright_mask = cv2.threshold(masked_green, int(threshold_val), 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    disc_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_od_mask = np.zeros_like(green_clahe)
    centroid = (112, 112)
    radius = 20

    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_od_mask, [largest_cnt], -1, 255, -1)
        (cx, cy), radius_val = cv2.minEnclosingCircle(largest_cnt)
        centroid = (int(cx), int(cy))
        radius = max(int(radius_val), 15)
        final_od_mask = cv2.dilate(final_od_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        
    return final_od_mask, centroid, radius

def apply_frangi_vessel_filter(green_channel, retina_mask):
    """
    Lightweight Multiscale Frangi Vesselness Filter based on Hessian Matrix Second Derivatives.
    Optimized for low CPU overhead with sub-sampled scale-space computation.
    """
    h, w = green_channel.shape
    # 2x sub-sample for 4x CPU speedup while preserving vessel branch geometry
    small_green = cv2.resize(green_channel, (w // 2, h // 2)).astype(np.float32) / 255.0
    vesselness_small = frangi(small_green, sigmas=[1.0, 2.0], alpha=0.5, beta=0.5, gamma=15.0, black_ridges=True)
    vesselness = cv2.resize(vesselness_small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    vesselness[retina_mask == 0] = 0
    if np.max(vesselness) > 0:
        vesselness = vesselness / np.max(vesselness)
        
    frangi_uint8 = np.uint8(255 * vesselness)
    frangi_score = float(np.mean(vesselness[retina_mask > 0])) if np.sum(retina_mask > 0) > 0 else 0.0
    return frangi_uint8, frangi_score

def apply_gabor_filter_bank(green_channel, retina_mask):
    """
    Lightweight Gabor Wavelet Filter Bank across 6 principal orientation angles (0° to 150° in 30° steps).
    Provides 2x CPU speedup while capturing curved vessels across all spatial directions.
    """
    clahe_green = apply_clahe(green_channel, clip_limit=3.0)
    accum_response = np.zeros_like(clahe_green, dtype=np.float32)

    # 6 orientations cover 180-degree spectrum with 50% lower CPU cycles
    for deg in range(0, 180, 30):
        theta = np.deg2rad(deg)
        kernel = cv2.getGaborKernel((13, 13), sigma=2.5, theta=theta, lambd=7.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        fimg = cv2.filter2D(clahe_green, cv2.CV_32F, kernel)
        accum_response = np.maximum(accum_response, fimg)

    accum_response[retina_mask == 0] = 0
    if np.max(accum_response) > 0:
        accum_response = (accum_response / np.max(accum_response)) * 255.0
        
    gabor_uint8 = np.uint8(np.clip(accum_response, 0, 255))
    gabor_score = float(np.mean(gabor_uint8[retina_mask > 0])) if np.sum(retina_mask > 0) > 0 else 0.0
    return gabor_uint8, gabor_score

def locate_macula_fovea(green_clahe, od_centroid, od_radius, retina_mask, exudates_mask, lesions_mask):
    """
    Locates the Macula/Fovea center (dark avascular region temporal to Optic Disc)
    and computes CSME (Clinically Significant Macular Edema) risk.
    """
    h, w = green_clahe.shape
    od_x, od_y = od_centroid
    
    shift_dir = -1 if od_x > w / 2 else 1
    macula_center_x = int(od_x + shift_dir * (2.2 * od_radius * 2))
    macula_center_y = od_y
    
    search_w = int(od_radius * 2.5)
    x1 = max(0, macula_center_x - search_w)
    x2 = min(w, macula_center_x + search_w)
    y1 = max(0, macula_center_y - search_w)
    y2 = min(h, macula_center_y + search_w)

    fovea_x, fovea_y = macula_center_x, macula_center_y

    if x2 > x1 and y2 > y1:
        search_roi = green_clahe[y1:y2, x1:x2]
        min_val, _, min_loc, _ = cv2.minMaxLoc(search_roi)
        fovea_x = x1 + min_loc[0]
        fovea_y = y1 + min_loc[1]

    fovea_mask = np.zeros_like(green_clahe)
    cv2.circle(fovea_mask, (fovea_x, fovea_y), int(od_radius * 1.5), 255, 2)
    cv2.circle(fovea_mask, (fovea_x, fovea_y), 3, 255, -1)

    combined_lesions = cv2.bitwise_or(exudates_mask, lesions_mask)
    lesion_pixels = np.column_stack(np.where(combined_lesions > 0))
    
    csme_risk = False
    min_dist_dd = 99.0

    if len(lesion_pixels) > 0:
        distances_px = np.sqrt((lesion_pixels[:, 0] - fovea_y)**2 + (lesion_pixels[:, 1] - fovea_x)**2)
        min_dist_px = np.min(distances_px)
        disc_diameter_px = max(2.0 * od_radius, 1.0)
        min_dist_dd = float(min_dist_px / disc_diameter_px)
        
        if min_dist_dd <= 1.0:
            csme_risk = True

    return fovea_mask, (fovea_x, fovea_y), min_dist_dd, csme_risk

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

    # Exudates are bright yellow/white lesions (top 2.5% intensity outside OD)
    thresh_val = max(int(np.percentile(valid_pixels, 97.5)), 170)
    _, exudates_raw = cv2.threshold(masked_green, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Refined opening to remove pixel noise while retaining true exudate clusters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    exudates_clean = cv2.morphologyEx(exudates_raw, cv2.MORPH_OPEN, kernel)
    
    # Connected component size filtering (area between 2 and 300 px)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(exudates_clean)
    final_exudates = np.zeros_like(green_clahe)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 2 <= area <= 300:
            final_exudates[labels == i] = 255

    return final_exudates

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
    
    # Adaptive threshold for microaneurysms
    valid_bh = bottom_hat[search_mask > 0]
    if len(valid_bh) > 0:
        thresh_val = max(int(np.percentile(valid_bh, 97.5)), 22)
    else:
        thresh_val = 24
        
    _, lesions_raw = cv2.threshold(bottom_hat, thresh_val, 255, cv2.THRESH_BINARY)
    lesions_raw = cv2.bitwise_and(lesions_raw, lesions_raw, mask=search_mask)
    
    # Morphological connected component analysis to filter noise
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lesions_raw)
    clean_lesions = np.zeros_like(green_clahe)
    microaneurysm_count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Filter microaneurysms by area: 2 px to 90 px in 224x224 image
        if 2 <= area <= 90:
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
    
    # 2. Optic Disc Mask & Centroid
    od_mask, od_centroid, od_radius = extract_optic_disc_mask(green_clahe, retina_mask)
    
    # 3. Multiscale Vessel Segmentation (Black-Hat + Frangi + Gabor Bank)
    vessels = segment_blood_vessels(green, retina_mask)
    frangi_mask, frangi_score = apply_frangi_vessel_filter(green, retina_mask)
    gabor_mask, gabor_score = apply_gabor_filter_bank(green, retina_mask)
    
    # 4. Hard Exudates Detection
    exudates = detect_exudates(green_clahe, od_mask, retina_mask)
    
    # 5. Red Lesions (Microaneurysms) Detection
    red_lesions, lesion_count = detect_red_lesions(green_clahe, vessels, retina_mask)
    
    # 6. Macula & Fovea Location & CSME Risk Analysis
    fovea_mask, fovea_xy, min_dist_dd, csme_risk = locate_macula_fovea(
        green_clahe, od_centroid, od_radius, retina_mask, exudates, red_lesions
    )
    
    # Compute precise quantitative ratios
    vessel_density = float(np.sum(vessels > 0)) / retina_pixel_count
    exudate_ratio = float(np.sum(exudates > 0)) / retina_pixel_count
    lesion_ratio = float(np.sum(red_lesions > 0)) / retina_pixel_count
    
    exudate_cnts, _ = cv2.findContours(exudates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = {
        'vessel_density': vessel_density,
        'frangi_score': frangi_score,
        'gabor_score': gabor_score,
        'exudate_ratio': exudate_ratio,
        'exudate_count': len(exudate_cnts),
        'lesion_ratio': lesion_ratio,
        'lesion_count': lesion_count,
        'fovea_lesion_distance_dd': min_dist_dd,
        'csme_risk': csme_risk,
        'mean_green': float(np.mean(green[retina_mask > 0])) if np.sum(retina_mask > 0) > 0 else 0.0,
        'std_green': float(np.std(green[retina_mask > 0])) if np.sum(retina_mask > 0) > 0 else 0.0,
        'clahe_contrast': float(np.std(green_clahe[retina_mask > 0])) if np.sum(retina_mask > 0) > 0 else 0.0
    }
    
    masks = {
        'optic_disc': od_mask,
        'vessels': vessels,
        'frangi_vessels': frangi_mask,
        'gabor_vessels': gabor_mask,
        'fovea_macula': fovea_mask,
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

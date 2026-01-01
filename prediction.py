import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import joblib

# --- CONFIGURATION ---
RF_MODEL_PATH = "/workspace/data/soildepth/Hypertune/TwoStageFusion/censored_1.0000_20251223_130932_fold4/binary_model.pkl"
QRF_MODEL_PATH = "/workspace/data/soildepth/Hypertune/TwoStageFusion/censored_1.0000_20251223_130932_fold4/qrf_model.pkl"

INDICES_DIR = "/workspace/data/soildepth/Indices/Ref_krycklan_Dataset/indices"
COMPOSITE_DIR = "/workspace/data/soildepth/Indices/Ref_krycklan_Dataset/CompositeBands"

# Output Directories
OUTPUT_ROOT = "/workspace/data/soildepth/Indices/Ref_krycklan_Dataset/Output"
DIRS = {
    "rf_prob": os.path.join(OUTPUT_ROOT, "RF_Probability"),
    "qrf_05":  os.path.join(OUTPUT_ROOT, "QRF_05"),
    "qrf_50":  os.path.join(OUTPUT_ROOT, "QRF_50_Median"),
    "qrf_95":  os.path.join(OUTPUT_ROOT, "QRF_95"),
    "fused":   os.path.join(OUTPUT_ROOT, "Fused_Result_10m")
}

PROB_THRESHOLD = 0.4
SCALE_FACTOR = 100 
NODATA_VAL = -9999

# Band Mapping (1-based index from TIF -> Feature Name)
COMPOSITE_BAND_MAPPING = {
    1: 'DEM', 2: 'EAS1ha', 3: 'EAS10ha', 4: 'DI2m', 5: 'CVA',
    6: 'SDFS', 7: 'DFME', 8: 'Rugged', 11: 'HKDepth', 13: 'LandAge',
    14: 'MSRM', 15: 'E', 16: 'N', 17: 'MED'
}

# Folder Mapping (Folder Name -> Model Feature Name)
FOLDER_TO_MODEL_MAPPING = {
    'Aspect50':'Aspect20',
    'RelaveTopographicPositions20_20': 'RTP20_20',
    'RelaveTopographicPositions50_50': 'RTP50_50',
    'Distance to faultlines': 'DistanceToDeformation',
    'ProfileCurvature20':'ProCur20',
    'lithotectonic_0': 'tekt_n_0',
    'lithotectonic_Undre skollberggrunden_67': 'tekt_n_67',
    'lithotectonic_Mellersta skollberggrunden_68': 'tekt_n_68',
    'lithotectonic_Undre del av mellersta skollberggrunden_69': 'tekt_n_69',
    'lithotectonic_Särvskollan_70': 'tekt_n_70',
    'lithotectonic_Seveskollorna_72': 'tekt_n_72',
    'lithotectonic_Köliskollorna_79': 'tekt_n_79',
    'lithotectonic_Undre Seveskollan_82': 'tekt_n_82',
    'lithotectonic_Kaledoniderna_88': 'tekt_n_88',
    'lithotectonic_Svekokarelska orogenen_337': 'tekt_n_337',
    'lithotectonic_Postsvekokarelska proterozoiska bergarter_346': 'tekt_n_346',
    'lithotectonic_Blekinge-Bornholmsorogenen_368': 'tekt_n_368',
    'lithotectonic_Östra segmentet_380': 'tekt_n_380',
    'lithotectonic_Östra segmentet, övre enheten_387': 'tekt_n_387',
    'lithotectonic_Östra segmentet, mellersta enheten_388': 'tekt_n_388',
    'lithotectonic_Östra segmentet, undre enheten_389': 'tekt_n_389',
    'lithotectonic_Idefjordenterrängen_390': 'tekt_n_390',
    'lithotectonic_Neoproterozoiska och fanerozoiska plattformstä_394': 'tekt_n_394',
    'lithotectonic_Blaikskollan_1939': 'tekt_n_1939',
}

# ---------------------------------------------------------

def setup_directories():
    for key, path in DIRS.items():
        os.makedirs(path, exist_ok=True)
    print(f"Output directories verified in {OUTPUT_ROOT}")

def load_models():
    print("Loading models...")
    rf_model = joblib.load(RF_MODEL_PATH)
    qrf_model = joblib.load(QRF_MODEL_PATH)
    print("Models loaded successfully.")
    return rf_model, qrf_model

def get_feature_names(model):
    if hasattr(model, "feature_names_in_"): return model.feature_names_in_
    elif hasattr(model, "feature_names"): return model.feature_names
    return None

def predict_quantiles(model, X):
    X_arr = X.values if hasattr(X, "values") else X
    all_tree_preds = np.stack([tree.predict(X_arr) for tree in model.estimators_])
    results = {}
    results['q05'] = np.percentile(all_tree_preds, 5, axis=0)
    results['q50'] = np.percentile(all_tree_preds, 50, axis=0)
    results['q95'] = np.percentile(all_tree_preds, 95, axis=0)
    return results

def to_int16_array(data, nodata_mask):
    scaled = data.copy()
    scaled = scaled * SCALE_FACTOR
    scaled[np.isnan(scaled)] = NODATA_VAL
    if nodata_mask is not None:
        scaled[~nodata_mask] = NODATA_VAL
    scaled = np.clip(scaled, -32768, 32767)
    return scaled.astype(np.int16)

def save_raster_2m(data_array, profile, output_dir, tile_name, valid_indices, img_shape):
    n_pixels = img_shape[0] * img_shape[1]
    final_image = np.full(n_pixels, NODATA_VAL, dtype=np.int16)
    int_data = to_int16_array(data_array, None)
    final_image[valid_indices] = int_data
    final_image = final_image.reshape(img_shape)
    
    out_path = os.path.join(output_dir, tile_name)
    profile.update({
        'driver': 'GTiff', 'dtype': 'int16', 'count': 1, 
        'nodata': NODATA_VAL, 'compress': 'lzw', 'predictor': 2
    })
    
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(final_image, 1)

def save_resampled_10m(data_array, src_profile, output_dir, tile_name, valid_indices, img_shape):
    n_pixels = img_shape[0] * img_shape[1]
    image_2m = np.full(n_pixels, np.nan, dtype=np.float32)
    image_2m[valid_indices] = data_array
    image_2m = image_2m.reshape(img_shape)

    dst_transform = src_profile['transform'] * src_profile['transform'].scale(5, 5)
    new_width = src_profile['width'] // 5
    new_height = src_profile['height'] // 5
    
    image_10m = np.empty((new_height, new_width), dtype=np.float32)

    reproject(
        source=image_2m,
        destination=image_10m,
        src_transform=src_profile['transform'],
        src_crs=src_profile['crs'],
        dst_transform=dst_transform,
        dst_crs=src_profile['crs'],
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan
    )

    int_10m = to_int16_array(image_10m, ~np.isnan(image_10m))

    out_path = os.path.join(output_dir, tile_name)
    new_profile = src_profile.copy()
    new_profile.update({
        'driver': 'GTiff', 'dtype': 'int16', 'count': 1, 'nodata': NODATA_VAL,
        'width': new_width, 'height': new_height, 'transform': dst_transform,
        'compress': 'lzw', 'predictor': 2
    })
    
    with rasterio.open(out_path, 'w', **new_profile) as dst:
        dst.write(int_10m, 1)

def process_tile(tile_name, rf_model, qrf_model, expected_features):
    print(f"\n==================================================")
    print(f"Processing tile: {tile_name}")
    print(f"==================================================")
    
    data_dict = {}

    # --- 1. Read Inputs ---
    comp_path = os.path.join(COMPOSITE_DIR, tile_name)
    if not os.path.exists(comp_path):
        print(f"  [Error] Composite file missing: {comp_path}")
        return

    print("  -> Reading Composite Bands:")
    with rasterio.open(comp_path) as src:
        profile = src.profile
        img_shape = src.shape
        n_pixels = src.width * src.height
        
        # Read Reference Band for Mask
        ref_band = src.read(1).flatten()
        valid_mask = ~np.isnan(ref_band)
        if src.nodata is not None: valid_mask = valid_mask & (ref_band != src.nodata)

        for band_idx, feature_name in COMPOSITE_BAND_MAPPING.items():
            try:
                data_dict[feature_name] = src.read(band_idx).flatten()
            except IndexError:
                print(f"     [CRITICAL ERROR] Band index {band_idx} out of range")
                return

    # Read external indices
    subfolders = [f.name for f in os.scandir(INDICES_DIR) if f.is_dir() and f.name != "CompositeBands"]
    print(f"  -> Reading {len(subfolders)} external index folders...")
    
    count_found = 0
    count_skipped = 0
    skipped_list = []  # <--- NEW: List to store missing names
    
    for folder_name in subfolders:
        model_feature_name = FOLDER_TO_MODEL_MAPPING.get(folder_name, folder_name)
        file_path = os.path.join(INDICES_DIR, folder_name, tile_name)
        
        is_valid = False
        
        if os.path.exists(file_path):
            with rasterio.open(file_path) as src:
                # Check dimensions before reading
                if src.shape == img_shape:
                    data_dict[model_feature_name] = src.read(1).flatten()
                    is_valid = True
                    count_found += 1
                else:
                    print(f"     [WARNING] Shape Mismatch! {folder_name}: {src.shape} vs {img_shape}. Filling with zeros.")
                    # (Note: We treat shape mismatch as invalid/skipped)
                    
        if not is_valid:
            # Fill with zeros if missing OR if shape was wrong
            data_dict[model_feature_name] = np.zeros(n_pixels)
            count_skipped += 1
            skipped_list.append(folder_name)  # <--- NEW: Add name to list
            
    # --- NEW: Print the specific files that were missing ---
    print(f"     Found {count_found} valid files. Filled {count_skipped} with zeros.")
    if skipped_list:
        print(f"     [INFO] Zero-filled features: {skipped_list}")

    # --- 2. Predict ---
    print("  -> Building DataFrame...")
    try:
        df = pd.DataFrame(data_dict)
    except ValueError as e:
        print(f"  [CRITICAL ERROR] DataFrame construction failed: {e}")
        for k, v in data_dict.items():
            if len(v) != n_pixels:
                print(f"     -> Feature '{k}' has length {len(v)} (Expected {n_pixels})")
        return

    valid_indices = np.where(valid_mask)[0]
    df_valid = df.iloc[valid_indices].copy()
    
    if df_valid.empty:
        print("  [Warning] No valid pixels found. Skipping.")
        return

    if expected_features is not None:
        missing_cols = set(expected_features) - set(df_valid.columns)
        if missing_cols:
            print(f"  [CRITICAL ERROR] Missing features: {missing_cols}")
            return
        df_valid = df_valid[expected_features]

    print("  -> Predicting Binary RF...")
    rf_prob = rf_model.predict_proba(df_valid)[:, 1]
    
    print("  -> Predicting QRF Quantiles (05, 50, 95)...")
    qrf_results = predict_quantiles(qrf_model, df_valid)
    
    print("  -> Fusing Models...")
    fused_pred = np.where(rf_prob > PROB_THRESHOLD, 0, qrf_results['q50'])

    # --- 3. Save Outputs ---
    print("  -> Saving 2m Quantiles (INT16)...")
    save_raster_2m(rf_prob, profile, DIRS["rf_prob"], tile_name, valid_indices, img_shape)
    save_raster_2m(qrf_results['q05'], profile, DIRS["qrf_05"], tile_name, valid_indices, img_shape)
    save_raster_2m(qrf_results['q50'], profile, DIRS["qrf_50"], tile_name, valid_indices, img_shape)
    save_raster_2m(qrf_results['q95'], profile, DIRS["qrf_95"], tile_name, valid_indices, img_shape)
    
    print("  -> Resampling & Saving Fused Map 10m (INT16)...")
    save_resampled_10m(fused_pred, profile, DIRS["fused"], tile_name, valid_indices, img_shape)
    
    print(f"  -> Done with {tile_name}")
    
def main():
    setup_directories()
    rf, qrf = load_models()
    
    # --- PRINT EXPECTED FEATURES ---
    expected_features_rf = get_feature_names(rf)
    expected_features_qrf = get_feature_names(qrf)
    
    print("\n" + "="*50)
    print("MODEL FEATURE EXPECTATIONS")
    print("="*50)
    print(f"RF Model expects {len(expected_features_rf)} features:")
    print(list(expected_features_rf))
    print("-" * 30)
    print(f"QRF Model expects {len(expected_features_qrf)} features:")
    print(list(expected_features_qrf))
    print("="*50 + "\n")

    # Check for mismatch between models (optional safety check)
    if set(expected_features_rf) != set(expected_features_qrf):
        print("[WARNING] RF and QRF models were trained on different features!")
        diff = set(expected_features_rf) ^ set(expected_features_qrf)
        print(f"Difference: {diff}\n")

    # Continue with processing...
    tile_files = [f for f in os.listdir(COMPOSITE_DIR) if f.endswith('.tif')]
    if not tile_files:
        print("No tiles found.")
        return

    for tile in tile_files:
        try:
            process_tile(tile, rf, qrf, expected_features_rf)
        except Exception as e:
            print(f"FAILED {tile}: {e}")

if __name__ == "__main__":
    main()
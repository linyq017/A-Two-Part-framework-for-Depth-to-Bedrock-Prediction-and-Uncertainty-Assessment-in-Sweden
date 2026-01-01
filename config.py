"""Configuration and constants."""
import numpy as np

# Quantiles
QUANTILES_FULL = [0.005, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.995]
QUANTILES_REDUCED = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# Depth bins
DEPTH_BINS = [-0.01, 0, 2, 5, 10, 15, 20, 30, np.inf]
DEPTH_LABELS = ['exposed', '0-2m', '2-5m', '5-10m','10-15m','15-20m', '20-30m', '>30m']

# Data paths (Simplified)
# Now pointing directly to the specific fold files you are using
DATA_PATHS = {
    'train': '/workspace/data/soildepth/DepthData/csvs/censored_encoded_folds/fold4_train.csv',
    'test': '/workspace/data/soildepth/DepthData/csvs/censored_encoded_folds/fold4_test.csv'
}

# Prediction config
PREDICTION_CONFIG = {
    'weighted_quantile': True,
    'weighted_leaves': True, 
}

# Plot config
PLOT_CONFIG = {
    'dpi': 600,
    'figsize_metrics': (15, 10),
    'figsize_spatial': (12, 5.5),
}

# Feature columns
FEATURE_COLUMNS = ['N', 'E', 'RTP20_20', 'RTP50_50', 'Slope20', 'DEM','Aspect20', 'ProCur20',
                   'EAS1ha', 'EAS10ha', 'DI2m', 'CVA', 'SDFS', 'DFME', 'Rugged', 'HKDepth',
                   'LandAge', 'MSRM', 'MED', 'jbas_merged_grus', 'jbas_merged_hall',
                   'jbas_merged_isalvssediment', 'jbas_merged_lera', 'jbas_merged_moran',
                   'jbas_merged_sand','jbas_merged_torv', 'tekt_n_0', 'tekt_n_67', 'tekt_n_68',
                   'tekt_n_69', 'tekt_n_70', 'tekt_n_72', 'tekt_n_79', 'tekt_n_82',
                   'tekt_n_88', 'tekt_n_337', 'tekt_n_346', 'tekt_n_368', 'tekt_n_380',
                   'tekt_n_387', 'tekt_n_388', 'tekt_n_389', 'tekt_n_390', 'tekt_n_394',
                   'tekt_n_1939', 'Geomorphon_Flat', 'Geomorphon_Footslope',
                   'Geomorphon_Hollow(concave)', 'Geomorphon_Peak(summit)',
                   'Geomorphon_Pit(depression)', 'Geomorphon_Ridge', 'Geomorphon_Shoulder',
                   'Geomorphon_Slope', 'Geomorphon_Spur(convex)', 'Geomorphon_Valley',
                   'DistanceToDeformation']
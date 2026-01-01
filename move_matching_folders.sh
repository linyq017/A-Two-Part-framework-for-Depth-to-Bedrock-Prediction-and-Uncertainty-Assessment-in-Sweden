#!/bin/bash

# 1. Define Directories
SRC_DIR="/workspace/data/soildepth/Indices/indices2m"
DEST_DIR="/workspace/data/soildepth/Indices/newQRFpred"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

echo "Source: $SRC_DIR"
echo "Destination: $DEST_DIR"
echo "-------------------------------------"

# 2. Define Mappings (Key -> Actual Folder Name)
# We use an associative array to map the ID to the specific folder name on disk
declare -A FOLDER_MAP

# --- Lithotectonic Mappings ---
FOLDER_MAP["tekt_n_0"]="lithotectonic_0"
FOLDER_MAP["tekt_n_67"]="lithotectonic_Undre skollberggrunden_67"
FOLDER_MAP["tekt_n_68"]="lithotectonic_Mellersta skollberggrunden_68"
FOLDER_MAP["tekt_n_69"]="lithotectonic_Undre del av mellersta skollberggrunden_69"
FOLDER_MAP["tekt_n_70"]="lithotectonic_Särvskollan_70"
FOLDER_MAP["tekt_n_72"]="lithotectonic_Seveskollorna_72"
FOLDER_MAP["tekt_n_79"]="lithotectonic_Köliskollorna_79"
FOLDER_MAP["tekt_n_82"]="lithotectonic_Undre Seveskollan_82"
FOLDER_MAP["tekt_n_88"]="lithotectonic_Kaledoniderna_88"
FOLDER_MAP["tekt_n_337"]="lithotectonic_Svekokarelska orogenen_337"
FOLDER_MAP["tekt_n_346"]="lithotectonic_Postsvekokarelska proterozoiska bergarter_346"
FOLDER_MAP["tekt_n_368"]="lithotectonic_Blekinge-Bornholmsorogenen_368"
FOLDER_MAP["tekt_n_380"]="lithotectonic_Östra segmentet_380"
FOLDER_MAP["tekt_n_387"]="lithotectonic_Östra segmentet, övre enheten_387"
FOLDER_MAP["tekt_n_388"]="lithotectonic_Östra segmentet, mellersta enheten_388"
FOLDER_MAP["tekt_n_389"]="lithotectonic_Östra segmentet, undre enheten_389"
FOLDER_MAP["tekt_n_390"]="lithotectonic_Idefjordenterrängen_390"
FOLDER_MAP["tekt_n_394"]="lithotectonic_Neoproterozoiska och fanerozoiska plattformstä_394"
FOLDER_MAP["tekt_n_1939"]="lithotectonic_Blaikskollan_1939"
FOLDER_MAP["ProCur20"]="ProfileCurvature20"

# --- Other Mappings ---
# FOLDER_MAP["E"]="Easting"
# FOLDER_MAP["N"]="Northing"
FOLDER_MAP["RTP20_20"]="RelaveTopographicPositions20_20"
FOLDER_MAP["RTP50_50"]="RelaveTopographicPositions50_50"
FOLDER_MAP["DistanceToDeformation"]="Distance to faultlines"


# 3. List of all Targets (Keys)
# These are the IDs you want to look for. 
# If a key is in FOLDER_MAP, we use the mapped value. If not, we use the key itself.
TARGETS=( 
    "RTP20_20" "RTP50_50" "Slope20"
    "jbas_merged_grus" "jbas_merged_hall" "jbas_merged_isalvssediment" 
    "jbas_merged_lera" "jbas_merged_moran" "jbas_merged_sand" "jbas_merged_torv" 
    "tekt_n_0" "tekt_n_67" "tekt_n_68" "tekt_n_69" "tekt_n_70" 
    "tekt_n_72" "tekt_n_79" "tekt_n_82" "tekt_n_88" "tekt_n_337" 
    "tekt_n_346" "tekt_n_368" "tekt_n_380" "tekt_n_387" "tekt_n_388" 
    "tekt_n_389" "tekt_n_390" "tekt_n_394" "tekt_n_1939" 
    "Geomorphon_Flat" "Geomorphon_Footslope" "Geomorphon_Hollow(concave)" 
    "Geomorphon_Peak(summit)" "Geomorphon_Pit(depression)" "Geomorphon_Ridge" 
    "Geomorphon_Shoulder" "Geomorphon_Slope" "Geomorphon_Spur(convex)" 
    "Geomorphon_Valley" "DistanceToDeformation" "Aspect20" "ProCur20" 
)
#    "DEM" "EAS1ha" "EAS10ha" "DI2m" "CVA" "SDFS" "DFME" "Rugged" "HKDepth" "LandAge" "MSRM" "MED" "N" "E" 
# 4. Loop and Move
for KEY in "${TARGETS[@]}"; do
    
    # Check if a mapping exists for this key
    if [[ -v FOLDER_MAP["$KEY"] ]]; then
        ACTUAL_NAME="${FOLDER_MAP[$KEY]}"
    else
        # If no mapping, assume folder name is same as key
        ACTUAL_NAME="$KEY"
    fi

    FULL_PATH="$SRC_DIR/$ACTUAL_NAME"

    # Check if the folder exists
    if [ -d "$FULL_PATH" ]; then
        echo "Found: '$ACTUAL_NAME' -> Moving..."
        mv "$FULL_PATH" "$DEST_DIR/"
    else
        echo "MISSING: Could not find folder '$ACTUAL_NAME' (ID: $KEY) in source."
    fi
done

echo "-------------------------------------"
echo "Operation complete."
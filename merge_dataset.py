import os
import shutil
import yaml
import uuid  # Added this to generate short names
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIGURATION SECTION
# ==========================================

TARGET_CLASSES = ['ambulance', 'police', 'firetruck', 'car']

DATASETS_CONFIG = {
    # Keep your path exactly as it was
    r'C:\Users\PAVAN\Desktop\MINIPROJECT\dasets\EVP.v1i.yolov8': {
        1: 0,   # 'Ambulance'
        10: 0,  # 'amblance2 - v2 blur_06_18'
        11: 0,  # 'ambulance'
        12: 0,  # 'ambulance_on'
        13: 0,  # 'ambulence' (typo in source)
        5: 1,   # 'Police_car'
        2: 2,   # 'Fire_truck'
        15: 2,  # 'fire truck'
        16: 2,  # 'firetruck_off'
        4: 3,   # 'Normal Vehicle'
        14: 3   # 'car'
    }
}

OUTPUT_DIR = Path('./unified_dataset')

# ==========================================
# EXECUTION SCRIPT
# ==========================================

def setup_directories():
    for split in ['train', 'valid', 'test']:
        for dtype in ['images', 'labels']:
            (OUTPUT_DIR / split / dtype).mkdir(parents=True, exist_ok=True)

def process_dataset(base_path, mapping, dataset_name):
    base_path = Path(base_path)
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        source_split_path = base_path / split
        if not source_split_path.exists() and split == 'valid':
            source_split_path = base_path / 'val'
        
        if not source_split_path.exists():
            continue

        img_dir = source_split_path / 'images'
        lbl_dir = source_split_path / 'labels'

        if not img_dir.exists(): 
            continue

        print(f"Processing {dataset_name} [{split}]...")
        
        # Enumerate gives us a counter (idx) to help with naming if needed, 
        # but we will use UUID for safety.
        for img_file in tqdm(list(img_dir.glob('*.*'))):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue

            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                continue 

            new_lines = []
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue
                    try:
                        source_id = int(parts[0])
                    except ValueError:
                        continue
                    
                    if source_id in mapping:
                        target_id = mapping[source_id]
                        new_line = f"{target_id} {' '.join(parts[1:])}\n"
                        new_lines.append(new_line)

            if new_lines:
                # --- FIX: GENERATE SHORT NAME ---
                # We use a random 8-char ID to keep the filename short and unique
                # Format: dsname_split_a1b2c3d4.jpg
                short_id = uuid.uuid4().hex[:8]
                new_stem = f"{dataset_name}_{split}_{short_id}"
                
                unique_filename = f"{new_stem}{img_file.suffix}"
                unique_labelname = f"{new_stem}.txt"
                
                # Copy Image with NEW SHORT NAME
                shutil.copy(img_file, OUTPUT_DIR / split / 'images' / unique_filename)
                
                # Write New Label File with SAME SHORT NAME
                with open(OUTPUT_DIR / split / 'labels' / unique_labelname, 'w') as out_f:
                    out_f.writelines(new_lines)

def create_yaml():
    yaml_content = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(TARGET_CLASSES),
        'names': TARGET_CLASSES
    }
    
    with open(OUTPUT_DIR / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"\nSuccessfully created data.yaml at {OUTPUT_DIR / 'data.yaml'}")

if __name__ == "__main__":
    if OUTPUT_DIR.exists():
        print(f"Cleaning existing directory: {OUTPUT_DIR}")
        try:
            shutil.rmtree(OUTPUT_DIR)
        except Exception as e:
            print(f"Error cleaning dir: {e}. Please close any open files and try again.")
            exit()
            
    setup_directories()
    
    for path, mapping in DATASETS_CONFIG.items():
        if os.path.exists(path):
            ds_name = os.path.basename(os.path.normpath(path))
            # Just in case the folder name itself is huge, slice it
            if len(ds_name) > 15:
                ds_name = ds_name[:15]
            process_dataset(path, mapping, ds_name)
        else:
            print(f"Error: Dataset path not found: {path}")

    create_yaml()
    print("\nMerge Complete. filenames have been shortened. Ready to train.")
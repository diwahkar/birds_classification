import os

import pandas as pd
from tqdm import tqdm

from constants import DATA_ROOT_DIR, NUM_CLASSES, LABELS


class BirdsDataPreparer:
    def __init__(self, root_dir=DATA_ROOT_DIR, num_classes=NUM_CLASSES):
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.labels = LABELS
        self.image_paths = {label: os.path.join(root_dir, label) for label in self.labels}

    def create_image_list_csv(self, output_csv='image_data.csv'):
        all_image_data = []

        for label_idx, label_name in enumerate(self.labels):
            current_dir = self.image_paths[label_name]
            if not os.path.exists(current_dir):
                print(f"Warning: Directory {current_dir} does not exist. Skipping.")
                continue

            files_in_current_dir = os.listdir(current_dir)

            for file_name in tqdm(files_in_current_dir, desc=f'Scanning {label_name} images...'):
                file_path = os.path.join(current_dir, file_name)
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    all_image_data.append({
                        'filepath': file_path,
                        'label': label_name,
                        'label_idx': label_idx
                    })

        if not all_image_data:
            print(f"No image data found in {self.root_dir}. Please ensure images are structured correctly.")
            return

        df = pd.DataFrame(all_image_data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.to_csv(output_csv, index=False)
        print(f"Image data saved to {output_csv} with {len(df)} entries.")
        return df

if __name__ == "__main__":
    preparer = BirdsDataPreparer()
    preparer.create_image_list_csv()
    print("\nData preparation complete. 'image_data.csv' created.")

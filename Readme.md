**How to Use:**

1.  **Setup:**
    * Create the project directory structure (`melanoma_classification/`, `data/`, etc.).
    * Place `HAM10000_metadata.csv` in the `data/` folder.
    * Place the HAM10000 image folders (e.g., `HAM10000_images_part_1`, `HAM10000_images_part_2`). **Important:** Adjust the `get_image_path` function in `dataset.py` if your image folder structure is different.
    * Install the dependencies: `pip install -r requirements.txt`

2.  **Configure:**
    * Edit `src/config.py` to set parameters like `MODEL_TYPE` ('CNN', 'ViT', 'DINO'), `EPOCHS`, `BATCH_SIZE`, `IMG_HEIGHT`, `IMG_WIDTH`, `USE_KFOLD`, `N_SPLITS`, etc.

3.  **Run Training:**
    * Open your terminal or command prompt.
    * Navigate to the `melanoma_classification/` directory.
    * Run the main training script: `python src/train.py`

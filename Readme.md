# Dermo: Skin Lesion Classification

A deep learning project for skin lesion classification using the HAM10000 dataset. The model can be configured for either binary (malignant vs benign) or multiclass (7 lesion types) classification.


## 📋 Requirements

- Python 3.8+
- TensorFlow 2.8+
- Additional dependencies:
  ```sh
  tensorflow
  pandas
  scikit-learn
  matplotlib
  tensorflow-hub
  transformers
  ```

## 🛠️ Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/dermo.git
   cd dermo
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Download the HAM10000 dataset
   - Create a `dataset` directory with the following structure:
     ```
     dataset/
     └── ham10000/
         ├── HAM10000_metadata.csv
         ├── HAM10000_images_part_1/
         └── HAM10000_images_part_2/
     ```

## ⚙️ Configuration

Edit `src/config.py` to customize:
- Model architecture (`MODEL_TYPE`)
- Classification type (`CLASSIFICATION_TYPE`: 'binary' or 'multiclass')
- Training parameters (`BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`)
- Data augmentation settings
- K-fold cross-validation options

## 🏃 Running the Project

1. (Optional) Generate augmented images:
   ```sh
   python src/augment_data.py
   ```

2. Train the model:
   ```sh
   python src/train.py
   ```

## 📊 Output

The training process generates:
- Trained models saved in `trained_models/`
- Training history plots in `logs/`
- ROC curves and confusion matrices
- Detailed evaluation metrics for each fold (if using K-fold)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- HAM10000 dataset: [The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions](https://doi.org/10.1038/sdata.2018.161)
- Based on architectures from TensorFlow and Hugging Face Transformers

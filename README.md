# Image-Caption-Generation-AI

This project implements an **Image Caption Generation AI** using deep learning techniques. It uses a CNN-RNN hybrid architecture to extract features from images and generate descriptive textual captions.


## 📁 Repository Structure
Image-Caption-Generation-AI/

├── Caption Generation Training/ # Training scripts and preprocessing

├── Caption Generation Testing/ # Inference and testing scripts

├── Dataset/ # Dataset and annotation files

├── Model Files/ # Trained model files (.h5, .pkl)

├── .gitattributes # Git LFS tracking

└── README.md # Project documentation

## 🧠 Model Overview

- **Architecture:** CNN (InceptionV3 / ResNet50) + RNN (LSTM)
- **Input:** Image
- **Output:** Natural language caption
- **Training Framework:** TensorFlow / Keras
- **Model Files:** Stored in `Model Files/` (use Git LFS for large files)


## ⚙️ Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/Hamza2-2/Image-Caption-Generation-AI.git
cd Image-Caption-Generation-AI
```
2. Install Dependencies
```
pip install tensorflow opencv-python numpy pandas matplotlib pillow
```

3. Download Git LFS Files
If not already done, install Git LFS and pull large files:
```
git lfs install
git lfs pull
```

## 🚀 How to Use

🏋️‍♂️ Train the Model
```
cd "Caption Generation Training"
python train_model.py
```
Make sure the dataset path is correctly set in the script.

🔍 Test / Generate Captions
```
cd "Caption Generation Testing"
python generate_caption.py --image_path path/to/image.jpg
```
The output will be the generated caption for the input image.

## 🗃️ Dataset

Dataset files and annotation formats are located in the Dataset/ directory. Ensure you place your images and captions properly before training.

## 📌 Notes

Model weights and tokenizer are stored in Model Files/

Scripts are modular and can be extended for GUI or web deployment

Training time depends on dataset size and hardware

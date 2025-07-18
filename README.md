# Image-Caption-Generation-AI

This project implements an **Image Caption Generation AI** using deep learning techniques. It uses a CNN-RNN hybrid architecture to extract features from images and generate descriptive textual captions.

## 🧠 How It Works

1. **Feature Extraction (CNN):**
   - A pretrained CNN (e.g., InceptionV3 or ResNet50) is used to extract high-level image features.
   - The output is a feature vector representing the image content.

2. **Sequence Modeling (RNN with LSTM):**
   - The image features are passed to an LSTM decoder along with previously generated words.
   - The model generates one word at a time, forming a complete caption.

3. **Tokenizer & Vocabulary:**
   - A tokenizer maps words to numeric indices and vice versa.
   - Start (`<start>`) and end (`<end>`) tokens define the boundaries of each caption.

4. **Training:**
   - Uses a paired dataset of images and corresponding captions.
   - Optimizes a sequence-to-sequence loss function to generate accurate descriptions.

5. **Inference:**
   - During testing, the model takes a new image and generates a caption by predicting the next word iteratively until the end token is reached.

## 💡 Real-World Applications

- 🔎 **Search Engines:** Indexing and retrieval of images based on captions
- 🧑‍🦯 **Accessibility:** Helping visually impaired users understand image content
- 📰 **Media Automation:** Auto-generating captions for news, blogs, and social media posts
- 🛍️ **E-Commerce:** Tagging and describing product images automatically

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

## Screenshots
<img width="312" height="477" alt="image" src="https://github.com/user-attachments/assets/9b1d7e1f-f341-4e07-aaa2-dd07669dba33" />


## 📌 Notes

- Model weights and tokenizer are stored in Model Files/

- Scripts are modular and can be extended for GUI or web deployment

- Training time depends on dataset size and hardware

## 👨‍💻 Developer

Hamza Afzal

BSCS, Bahria University, Islamabad

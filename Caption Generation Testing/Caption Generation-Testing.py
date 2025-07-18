import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

# Paths 
DATA_PATH = 'Dataset'
MODEL_PATH = 'Model Files/Caption-Generation_Model.h5'
CAPTIONS_FILE = os.path.join(DATA_PATH, 'captions.txt')   


#Feature Extraction Method
def extract_features(img_path):
    base_model = VGG16()
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

    image = load_img(img_path, target_size=(224, 224))
    img_display = image.copy()
    image = img_to_array(image)
    image = image.reshape((1, 224, 224, 3))
    image = preprocess_input(image)

    feature = model.predict(image, verbose=0)
    return feature, img_display


# Load and prepare Captions
def load_and_clean_captions(file_path):
    with open(file_path, 'r') as f:
        next(f)
        lines = f.read().strip().split('\n')

    mapping = {}
    for line in lines:
        tokens = line.split(',')
        if len(tokens) < 2:
            continue
        image_id, caption = tokens[0].split('.')[0], " ".join(tokens[1:])
        caption = caption.lower()
        caption = ' '.join([word for word in caption.split() if len(word) > 1])
        caption = f'start {caption} end'
        mapping.setdefault(image_id, []).append(caption)
    return mapping


# Tokenizer Method
def create_tokenizer(captions_mapping):
    all_captions = [caption for captions in captions_mapping.values() for caption in captions]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    max_len = max(len(caption.split()) for caption in all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, max_len, vocab_size


# Word prediction
def index_to_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None


# Caption Prediction
def predict_caption(model, image_features, tokenizer, max_length):
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y  = model.predict([image_features, sequence], verbose=0)
        predicted_idx = np.argmax(y)
        word = index_to_word(predicted_idx, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


# Tkinter GUI
class CaptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Caption Generator")
        self.root.geometry("600x500") 

        self.image_path = None
        self.tk_img = None

        Label(root, text="Upload an image and generate its caption", font=("Arial", 14)).pack(pady=15)

        self.img_label = Label(root)
        self.img_label.pack()

        self.upload_btn = Button(root, text="Upload Image", command=self.upload_image, font=("Arial", 11), width=20)
        self.upload_btn.pack(pady=8)

        self.caption_btn = Button(root, text="Generate Caption", command=self.generate_caption_gui, font=("Arial", 11), width=20)
        self.caption_btn.pack(pady=8)

        self.caption_label = Label(root, text="", font=("Arial", 13), wraplength=550, justify="center")
        self.caption_label.pack(pady=15)

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if self.image_path:
            pil_img = Image.open(self.image_path)
            pil_img_resized = pil_img.resize((350, 250))
            self.tk_img = ImageTk.PhotoImage(pil_img_resized)
            self.img_label.config(image=self.tk_img)
            self.caption_label.config(text="")

    def generate_caption_gui(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        try:
            image_feature, _ = extract_features(self.image_path)
            mapping = load_and_clean_captions(CAPTIONS_FILE)
            tokenizer, max_length, _ = create_tokenizer(mapping)
            model = load_model(MODEL_PATH, compile=False)
            model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

            caption = predict_caption(model, image_feature, tokenizer, max_length)
            caption = caption.replace("start", "").replace("end", "").strip()
            self.caption_label.config(text=f"Caption: {caption}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate caption.\n{e}")


# main function
if __name__ == "__main__":
    root = tk.Tk()
    app = CaptionApp(root)
    root.mainloop()

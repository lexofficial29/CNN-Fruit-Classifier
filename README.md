# 🍎 CNN Fruit Classifier

A PyTorch-based Convolutional Neural Network for classifying different types of fruits from images. This project supports training, testing, and batch prediction with a custom dataset organized in folders.

---

## 🧠 Model Architecture

- Built with 4 convolutional layers followed by ReLU activation and max pooling.  
- Fully connected classifier with dropout for regularization.  
- Uses AdamW optimizer and a step learning rate scheduler.  
- Final output layer dynamically adapts to the number of fruit classes.

---

## 📁 Folder Structure

(train/, test/, and predict/ folders)

- train/
  - apple/
  - banana/
  - ...
- test/
  - apple/
  - banana/
  - ...
- predict/
  - apple/
  - banana/
  - ...

Each subfolder represents a fruit class and contains images.

---

## ⚙️ Requirements

- Python 3.8+  
- PyTorch  
- torchvision  
- PIL (Pillow)

Install dependencies:

```
pip install torch torchvision pillow
```

---

## 🚀 Training

To train the model:

```
python train.py
```

This will:

- Train the CNN on `train/`
- Evaluate on `test/` and `predict/`
- Save the model as `classifier.pth`

You can configure:
- Epochs
- Learning rate
- Batch size
- Image dimensions

---

## 🔍 Evaluation & Prediction

To evaluate predictions and print per-class accuracy:

```
python test.py
```

This will:

- Load the trained model (`classifier.pth`)  
- Evaluate images in the `predict/` folder  
- Print predictions and per-class accuracy report  

Example output:

```
[apple] IMG_001.jpg → Predicted: apple  
[banana] IMG_021.jpg → Predicted: banana  
...

--- Classification Report ---
apple        | Accuracy: 95.00% (19/20)  
banana       | Accuracy: 90.00% (18/20)  
...  
Overall Accuracy: 92.50% (37/40)
```

---

## 🛠 Features

- 🖼 Image resizing, random flips & rotation for data augmentation  
- 🧠 Automatic class name detection  
- 🔁 Learning rate scheduler  
- 💾 Model saving  
- 📊 Per-class accuracy stats

---

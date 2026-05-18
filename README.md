# Flowers-Classification
## Overview
This project implements a Convolutional Neural Network (CNN) for classifying flower images into five categories: rose, sunflower, daisy, tulip, and dandelion using TensorFlow/Keras.  
Validation Accuracy: 67.21%
## Dataset
The used dataset can be found in Kaggle: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition  
It contains:  
- **Tulip:** 984 images  
- **Dandelion:** 1052 images  
- **Sunflower:** 733 images  
- **Rose:** 784 images  
- **Daisy:** 764 images  
<img width="650" height="120" alt="image" src="https://github.com/user-attachments/assets/3c87583e-8651-491b-9bfa-7499d7ae96df" />

- **Classes Distribution:**
<img width="300" height="250" alt="image" src="https://github.com/user-attachments/assets/f777804b-7949-43af-bcb0-533798436d1d" />



## Preprocessing
Images were resized to 128×128×3 and preprocessed using ImageDataGenerator with data augmentation techniques such as rotation, shifting, zooming, and horizontal flipping to improve model generalization.  
The dataset was automatically split into:  
- **Training data:** 3457 images  
- **Validation data:** 860 images  

## Model Architecture
The CNN architecture consists of:  
- 3 convolutional layers with batch normalization and max pooling  
- Dense layer with dropout regularization  
- Softmax output layer for 5-class classification  

## Results


## Run Prediction  
```bash
pip install -r requirements.txt
python main.py
```

## Future Improvements
- Use transfer learning (VGG16/ResNet) to improve accuracy  
- Increase dataset size and balance classes  
- Build a simple web app for image prediction  
- Apply advanced data augmentation techniques  

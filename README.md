# Emotion Images

Real-time webcam emotion detection that shows one of your own images for the detected emotion.

## Features

- Detects your face and classifies its emotion in real time using a pre-trained model.
- Displays an image corresponding to the detected emotion (happy, sad, angry, etc.).
- Works with either a conda environment (`environment.yml`) or a standard virtual environment using `requirements.txt`.  
- Uses your own photos stored in the `images/` folder.

## Setup with Conda (Recommended):
```bash
git clone https://github.com/JacksonByrne/Emotion-to-Reaction-Image.git
cd Emotion-to-Reaction-Image
conda env create -f emotion_detect.yml
conda activate emotion_detect
```

## Or Setup with pip / venv
### 1. Install PyTorch

PyTorch needs a platform-specific wheel, so install it using the official command for your OS and Python version:

https://pytorch.org/get-started/locally/

Example (CPU only):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install the libraries required
```bash
git clone https://github.com/JacksonByrne/Emotion-to-Reaction-Image.git
cd Emotion-to-Reaction-Image

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```
(For windows use .venv\Scripts\activate instead of .venv/bin/activate)

## To Run:
```bash
cd src
python face_to_images.py
```

## To change images:
Go to 'images' folder and change images to whatever you'd like. Note that images must be named 'happy.jpg' etc.  
Images must also be .jpg ONLY

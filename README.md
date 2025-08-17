# MED VQA

## 📋 Table of Contents
1. [About MED VQA](#-about-med-vqa)
2. [Setup Instructions](#-setup-instructions)
3. [Demo Video](#-demo-video)
4. [Model Training Instructions](#-model-training-instructions)
5. [Contributors](#-contributors)

## 🩺 About MED VQA

**MED VQA** is a Visual Question Answering system tailored for the medical domain, supporting both radiology and pathology images. Users can upload an image, ask questions, and receive accurate AI-generated responses using fine-tuned transformer models.

Key Features:
- Specialized models for Radiology and Pathology.
- Intuitive web-based interface.
- Custom-trained models on domain-specific datasets.

## 🛠️ Setup Instructions

### 1. Install Node.js

#### For Ubuntu/Debian:
```bash
sudo apt update
sudo apt install nodejs npm
```

#### For MacOS:
```bash
brew install node
```

#### Verify Installation:
```bash
node --version
npm --version
```

### 2. Create and Setup Conda Environment

1. **Create a new Conda environment**:
   ```bash
   conda create -n vqa_env python=3.10
   ```

2. **Activate the environment**:
   ```bash
   conda activate vqa_env
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Update Model File Paths

Update the file paths for the models in the following files:

- `server/src/models/pathology/inference.py`
- `server/src/models/radiology/inference.py`

Replace the file paths with the exact full paths of the saved model files in their respective directories:

- `final_med_vqa_rad_model.pt`
- `final_med_vqa_path_model.pt`

### 4. Running the Website

1. **Activate the Conda environment**:
   ```bash
   conda activate vqa_env
   ```

2. **Run the website**:
   ```bash
   npm start
   ```

### 5. Accessing the Website

Open your web browser and navigate to:

```bash
https://localhost:3000
```

This will open the VQA model interface.

## 🎥 Demo Video

[![Watch the Demo](https://img.youtube.com/vi/bFl-fEqAkVM/0.jpg)](https://youtu.be/bFl-fEqAkVM)

## 🧠 Model Training Instructions

1. **Activate the Conda environment**:
   ```bash
   conda activate vqa_env
   ```

2. **Navigate to the models folder**:
   ```bash
   cd server/src/models/
   ```

3. **Choose the respective folder**:
   - For Pathology: `cd pathology`
   - For Radiology: `cd radiology`

4. **Run the training script**:
   ```bash
   python main.py
   ```

## 👨‍💻 Contributors

- Abhijeet  
- Chaitanya  
- Devansh  
- Sohel  
- Tousif


## Prerequisites

- **Node.js**
- **Python**

---

## Setup Instructions

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

#### For Windows:
Download and install Node.js from [https://nodejs.org/](https://nodejs.org/).

#### Verify Installation:
```bash
node --version
npm --version
```

---

### 2. Create and Setup Conda Environment

1. **Create a new Conda environment**:
   ```bash
   conda create -n vqa_env python=3.8
   ```

2. **Activate the environment**:
   ```bash
   conda activate vqa_env
   ```

3. **Install required packages**:
   ```bash
   pip install -r server/src/models/pathology/requirements.txt

   pip install -r server/src/models/radiology/requirements.txt
   ```

---

### 3. Update Model File Paths

Update the file paths for the models in the following files:

- `server/src/models/pathology/inference.py`
- `server/src/models/radiology/inference.py`

Replace the file paths with the exact full paths of the saved model files in their respective directories:

- `final_med_vqa_rad_model.pt`
- `final_med_vqa_path_model.pt`

---

### 4. Running the Website

1. **Activate the Conda environment**:
   ```bash
   conda activate vqa_env
   ```

2. **Run the website**:
   ```bash
   npm start
   ```

---

### 5. Accessing the Website

Open your web browser and navigate to:

```bash
https://localhost:3000
```

This will open the VQA model interface.

---

### 6. Trial Models Drive Link

Download the trial models from the following Google Drive link:

[Trial Models Drive Link](https://drive.google.com/drive/folders/1LYL00BNCaEHwiYOCwUy-WWxvgsJpE1dy)

---

### 7. Model Training Instructions

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

---
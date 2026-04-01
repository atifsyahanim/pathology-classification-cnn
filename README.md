# pathology-classification-cnn
This repository contains two main components for my FYP project on **voice-based pathology classification**:

---

## 1. Prototype (`prototype/`)

- A Gradio-based web application demonstrating the **voice pathology classification model**.
- Includes:
  - `app.py` → Gradio interface
  - `model_resnet18.h5` → trained model
  - `requirements.txt` → Python dependencies
- Quick demo for recruiters or users to test the model.

**Run locally:**

```bash
cd prototype
pip install -r requirements.txt
python app.py
```

## 2. Full Notebook (`full_notebook/`)
Complete Colab notebook containing:
- Data preprocessing & augmentation
- Training of 30+ models with fine-tuning
- Evaluation and performance analysis
This is for anyone interested in reproducing the experiments or exploring the details.

## Data
- The dataset used for this project is **not included** due to privacy and ethical reasons.  
- Some data was collected from hospitals, and some is self-collected.  

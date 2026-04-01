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

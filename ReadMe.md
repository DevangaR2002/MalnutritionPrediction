# Hybrid Ensemble Model – Quick Run Guide

This guide explains **only** how to create a virtual environment, install dependencies, and run  
`HybridEnsembleModel.ipynb`.

---

## 1. Create Virtual Environment

```bash
python -m venv venv
````

---

## 2. Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure `requirements.txt` is in the same directory as the notebook.

---

## 4. Run the Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open and run **only**:

```
HybridEnsembleModel.ipynb
```

Run all cells **top to bottom**.

---

## 5. Output

After successful execution, the notebook will generate:

* Trained hybrid ensemble model (`.pkl`)
* Evaluation metrics and plots
* Comparison results (if included in the notebook)

---


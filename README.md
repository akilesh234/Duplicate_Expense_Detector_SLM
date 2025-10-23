# Duplicate Expense Detector

An AI-powered web-based system to automatically detect and manage duplicate expense records across multiple data formats using Machine Learning (ML) and Natural Language Processing (NLP).

---

## Overview

The **Duplicate Expense Detector** helps organizations identify duplicate expenses from receipts and financial reports.  
It uses Optical Character Recognition (OCR) to extract text from images and a fine-tuned Sentence Transformer model to compute similarity between expense descriptions.

This project was developed as a Mini Project by students of  
**Department of Artificial Intelligence and Data Science, PSG iTech, Coimbatore**,  
in collaboration with **SAP Labs, Bengaluru**.

---

## Key Features

- Multi-format input: supports images, CSV, and Excel files  
- AI-powered text similarity using a fine-tuned Sentence Transformer model  
- OCR integration with Tesseract for image-based receipts  
- Configurable similarity threshold for duplicate detection  
- React-based web interface for reviewing and managing duplicates  
- Secure data processing and privacy handling  

---

## Tech Stack

| Layer | Technology |
|--------|-------------|
| Frontend | React.js |
| Backend | Flask (Python) |
| NLP / Machine Learning | Sentence Transformer, TensorFlow |
| OCR | Tesseract |
| Storage | CSV (persistent) |
| API Architecture | RESTful APIs |

---

## System Workflow

```
+---------------------------+
|        React Frontend     |
|  (Upload, Review, UI)     |
+-------------+-------------+
              |
              ▼
+---------------------------+
|        Flask Backend      |
|  - File handling          |
|  - OCR & NLP processing   |
+-------------+-------------+
              |
              ▼
+---------------------------+
|   Sentence Transformer    |
| (Computes text similarity)|
+-------------+-------------+
              |
              ▼
+---------------------------+
|    Duplicate Results      |
| (Displayed on Frontend)   |
+---------------------------+
```

---

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/akilesh234/Duplicate_Expense_Detector_SLM.git
cd Duplicate_Expense_Detector_SLM
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install the required dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask backend
```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```

### 5. Run the React frontend
```bash
cd frontend
npm install
npm start
```

---

## Results and Performance

- Achieved **95% precision** and **92% recall** on the test dataset  
- Average processing time:  
  - Images: ~2.5 seconds per image  
  - CSV: ~0.5 seconds per 1000 entries  
  - Excel: ~0.7 seconds per 1000 entries  
- User feedback indicated high accuracy and ease of use  

---

## Limitations

- OCR accuracy depends on image quality  
- Large datasets may require higher computational resources  
- Currently focused on textual similarity; does not cover all semantic duplicates  

---

## Future Work

- Integration with ERP or accounting systems (e.g., SAP Concur, QuickBooks)  
- Advanced analytics and financial reporting modules  
- Improved OCR handling for low-quality or multi-language receipts  
- Continuous model retraining for enhanced accuracy  
- Development of a mobile version for expense submission and review  

---

## Contributors

- **Akilesh J S (715522243004)**  
- **Dharshini (715522243016)**  
- **Dhynesh J (715522243017)**  

**Industry Mentor:** Mr. Hirdyansh Sharma, SAP Labs, Bengaluru  
**Academic Mentor:** Dr. S. Lokesh, Associate Professor, PSG iTech  

---

## References

1. Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*  
2. Vaswani, A., et al. (2017). *Attention Is All You Need.*  
3. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*  
4. Smith, R. (2007). *An Overview of the Tesseract OCR Engine.*  
5. Abadi, M., et al. (2016). *TensorFlow: A System for Large-Scale Machine Learning.*  
6. React.js and Flask official documentation.  

---

### Repository Link
[https://github.com/akilesh234/Duplicate_Expense_Detector_SLM](https://github.com/akilesh234/Duplicate_Expense_Detector_SLM)

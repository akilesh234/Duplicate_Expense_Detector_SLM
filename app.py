from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import pandas as pd
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_PATH'] = 10000000  # 10MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CSV_FILE = 'extracted_data.csv'
model = SentenceTransformer('fine-tuned-model')

# Ensure pytesseract is configured correctly
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path as necessary

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_image(file_path):
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        data = {'description': [], 'amount': [], 'date': []}
        for line in text.split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    data['description'].append(' '.join(parts[:-2]))
                    data['amount'].append(parts[-2])
                    data['date'].append(parts[-1])
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        raise ValueError(f"Error extracting text from image: {e}")

def load_existing_data():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        return pd.DataFrame(columns=['description', 'amount', 'date'])

def save_data(df):
    if os.path.exists(CSV_FILE):
        existing_df = pd.read_csv(CSV_FILE)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

def identify_columns(df):
    description_col, amount_col, date_col = None, None, None
    for col in df.columns:
        if any(keyword in col for keyword in ['desc', 'description', 'category', 'item']):
            description_col = col
        elif any(keyword in col for keyword in ['amount', 'inr', 'cost', 'price', 'expense']):
            amount_col = col
        elif any(keyword in col for keyword in ['date', 'time', 'day']):
            date_col = col
    if not description_col or not amount_col or not date_col:
        raise ValueError("Could not identify columns. Please check the dataset.")
    return description_col, amount_col, date_col

def detect_duplicates_with_language_model(df, description_col, amount_col, date_col):
    df['combined'] = df[description_col].astype(str) + ' ' + df[amount_col].astype(str) + ' ' + df[date_col].astype(str)
    embeddings = model.encode(df['combined'], convert_to_tensor=True)
    embeddings = embeddings.cpu()
    cosine_sim_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()
    threshold = 0.85
    duplicates = []
    for i in range(len(cosine_sim_matrix)):
        for j in range(i + 1, len(cosine_sim_matrix)):
            if cosine_sim_matrix[i, j] > threshold:
                duplicates.append((i, j, cosine_sim_matrix[i, j]))
    return duplicates

def display_duplicates(df, duplicates):
    if not duplicates:
        return []

    duplicates = sorted(duplicates, key=lambda x: x[2], reverse=True)
    top_duplicates = duplicates[:10]

    results = []
    for dup in top_duplicates:
        result = {
            "First Entry": df.iloc[dup[0]].to_dict(),
            "Second Entry": df.iloc[dup[1]].to_dict(),
            "Similarity Score": dup[2]
        }
        results.append(result)
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            df = extract_text_from_image(file_path)
            
            existing_df = load_existing_data()
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            description_col, amount_col, date_col = identify_columns(combined_df)
            duplicates = detect_duplicates_with_language_model(combined_df, description_col, amount_col, date_col)
            results = display_duplicates(combined_df, duplicates)
            
            save_data(df)
            
            return render_template('results.html', results=results)
        except ValueError as e:
            flash(str(e))
            return redirect(url_for('index'))
    else:
        flash('Unsupported file format')
        return redirect(url_for('index'))

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

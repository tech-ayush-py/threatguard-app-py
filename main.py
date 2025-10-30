from flask import Flask, render_template, request
import google.generativeai as genai
import os
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import atexit
import json  # <-- Added for parsing Gemini's response
import json
from dotenv import load_dotenv  # <-- Import this

load_dotenv()  # <-- Add this line to load the .env file

app = Flask(__name__)

# Set up the Google API Key
# Now it safely loads from your .env file
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("--- Error: GOOGLE_API_KEY not found. Set it in your .env file. ---")
    # You might want to exit or handle this error
else:
    genai.configure(api_key=API_KEY)



# Initialize the Gemini model (for the email/text scam feature)
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
    print("--- Gemini model initialized successfully. ---")
except Exception as e:
    print(f"--- Error initializing Gemini model: {e} ---")
    model = None

# --- ML Model Training ---
url_vectorizer = None
url_model = None


def train_url_model():
    """
    Loads data from 'data.csv' and trains a
    Logistic Regression model.
    """
    global url_vectorizer, url_model
    try:
        # Load the dataset
        data = pd.read_csv('data.csv')
        data.dropna(inplace=True)

        if 'URL' not in data.columns or 'Label' not in data.columns:
            print("Error: CSV must have 'URL' and 'Label' columns.")
            return

        X = data['URL']
        y = data['Label']

        url_vectorizer = TfidfVectorizer()
        X_tfidf = url_vectorizer.fit_transform(X)

        url_model = LogisticRegression()
        url_model.fit(X_tfidf, y)

        print("--- URL detection model trained successfully. ---")

    except FileNotFoundError:
        print("--- Error: 'data.csv' not found. URL detection will not work. ---")
    except Exception as e:
        print(f"--- Error training URL model: {e} ---")


# --- END ML Model Training ---


# --- UPDATED HELPER FUNCTION ---
def predict_fake_or_real_email_content(text):
    """
    Analyzes text and returns a dictionary with 'class' and 'details'.
    """
    if not model:
        return {"class": "Error", "details": "Gemini model is not initialized."}

    # New prompt asking for a JSON response
    prompt = f"""
    Analyze the following text and determine if it is "Scam" or "Legitimate".
    Provide a brief "details" explaining your reasoning (max 2-3 sentences).
    Return your answer ONLY as a valid JSON object with two keys: "class" and "details".

    Example for a scam:
    {{"class": "Scam", "details": "This message creates false urgency, uses a suspicious link, and asks for a small fee to steal credit card details."}}

    Example for legitimate:
    {{"class": "Legitimate", "details": "This appears to be a standard order confirmation with no suspicious links or urgent requests."}}

    Text to analyze:
    ---
    {text}
    ---
    """
    try:
        response = model.generate_content(prompt)

        # Clean the response to find the JSON
        clean_text = response.text.strip().lstrip("```json").rstrip("```")

        # Parse the JSON string into a Python dictionary
        result_dict = json.loads(clean_text)
        return result_dict

    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        # Fallback in case JSON parsing fails
        return {"class": "Error", "details": "Failed to analyze content. The response was not valid JSON."}


# --- END UPDATED FUNCTION ---


# --- UPDATED URL DETECTION FUNCTION ---
def url_detection(url):
    """
    Classifies a URL as 'good' or 'bad' using the pre-trained ML model.
    """
    if not url_model or not url_vectorizer:
        return "model_not_ready"

    url_tfidf = url_vectorizer.transform([url])
    prediction = url_model.predict(url_tfidf)[0]

    # Map 'good'/'bad' to the categories your HTML expects
    if prediction == 'good':
        return 'benign'
    else:
        # Assuming 'bad' maps to 'phishing', 'malware', etc.
        return 'phishing'


# --- Routes ---

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/scam/', methods=['POST'])
def detect_scam():
    if 'file' not in request.files:
        return render_template("index.html", message="No file uploaded.")

    file = request.files['file']
    extracted_text = ""

    if file.filename.endswith('.pdf'):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            extracted_text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        except Exception as e:
            return render_template("index.html", message=f"Error reading PDF: {e}")
    elif file.filename.endswith('.txt'):
        extracted_text = file.read().decode("utf-8")
    else:
        return render_template("index.html", message="Invalid file type. Please upload a PDF or TXT file.")

    if not extracted_text.strip():
        return render_template("index.html", message="File is empty or text could not be extracted.")

    # --- MODIFIED ---
    # Call the new function which returns a dict
    result_dict = predict_fake_or_real_email_content(extracted_text)

    # Format the dict into a simple string for the 'message' variable
    # (since the file section's HTML expects a simple string)
    return render_template("index.html", file_analysis_result=result_dict)
    # --- END MODIFICATION ---


# --- NEW ROUTE ADDED ---
@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """
    Handles the text paste submission.
    """
    text_to_analyze = request.form.get('message_text', '').strip()

    if not text_to_analyze:
        return render_template("index.html",
                               text_analysis_result={"class": "Error", "details": "No text was provided."})

    # Call the helper function
    result_dict = predict_fake_or_real_email_content(text_to_analyze)

    # Pass the entire dictionary to the template
    return render_template("index.html", text_analysis_result=result_dict)


# --- END NEW ROUTE ---


@app.route('/predict', methods=['POST'])
def predict_url():
    url = request.form.get('url', '').strip()

    if not url:
        return render_template("index.html", url_error="No URL provided.", input_url=url)

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    classification = url_detection(url)

    if classification == "model_not_ready":
        return render_template("index.html",
                               url_error="URL detection model is not ready. Check 'data.csv'.",
                               input_url=url)

    return render_template("index.html", input_url=url, predicted_class=classification)


if __name__ == '__main__':
    # Train the model *before* starting the app
    train_url_model()
    app.run(debug=True)
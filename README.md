# ThreatGuard: Phishing & Scam Detector

This is a web application built with Flask and Python that uses machine learning and AI to detect malicious URLs, scam emails, and phishing attempts.

## Features

* **URL Analysis:** Uses a Logistic Regression model trained on a URL dataset (`data.csv`) to classify links as 'benign' or 'phishing'.
* **Text Analysis:** Uses the Google Gemini API to analyze pasted text (emails, messages) for scam tactics.
* **File Analysis:** Scans uploaded `.txt` or `.pdf` files for malicious content using the same Gemini API.

## How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/phishing-detector-app.git](https://github.com/YourUsername/phishing-detector-app.git)
    cd phishing-detector-app
    ```

2.  **Create a virtual environment and install packages:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Set up your API Key:**
    * Create a file named `.env`.
    * Inside, add your Google API key: `GOOGLE_API_KEY="YOUR_API_KEY_HERE"`

4.  **Run the app:**
    ```bash
    flask run
    ```

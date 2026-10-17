# fake-review-detector

Fake Review Detector
A demo web application that analyzes a customer review and estimates
whether it may be FAKE, GENUINE, or NEEDS_REVIEW. It
combines a small machine-learning model with rule-based signals,
rating--sentiment checks, image EXIF GPS metadata, duplicate-review
matching, and business review-history analysis.
Important: This is an educational prototype, not a
production-ready fraud detector. The classifier is trained on a very
small synthetic dataset, and its scores are heuristic rather than
calibrated probabilities. A suspicious signal does not prove that a
review is fake.

Table of Contents
- Features
- How It Works
- Tech Stack
- Project Structure
- Requirements
- Installation
- Run the Application
- Using the App
- API Reference
- Understanding the Results
- Current Limitations
- Possible Improvements
- Disclaimer
Features
- Text-based ML scoring: Uses TF-IDF text features and Logistic
  Regression.
- Rating and sentiment comparison: Checks whether the star rating
  conflicts with positive or negative words in the review.
- Suspicious text patterns: Detects promotional phrases, repeated
  words, excessive capitalization, excessive exclamation marks, and
  very short reviews.
- Abusive-language rule: Can flag reviews containing configured
  severe words or a threshold number of mild words.
- Image metadata inspection: Reads GPS coordinates from image EXIF
  metadata when available.
- Distance check: If business latitude and longitude are supplied
  to the API, compares the image GPS coordinates with the business
  location.
- Duplicate / near-duplicate check: Compares the submitted text
  with historical reviews using SequenceMatcher.
- Business-history analysis: Checks whether a review's rating
  opposes the majority rating pattern when sufficient history is
  available.
- Interactive interface: Streamlit frontend for uploading an
  image, entering a review, choosing a rating, and viewing the
  results.
- REST API: FastAPI backend exposes a /predict endpoint.
How It Works
1. The user uploads a JPG/JPEG or PNG image, enters review text, and
   selects a star rating.
2. The Streamlit frontend sends the form data and image to the FastAPI
   backend.
3. The backend validates the input and opens the image.
4. The ML model calculates a review score using TF-IDF text features,
   rating, sentiment, and text length.
5. Additional rules check text patterns, rating--sentiment mismatch,
   abusive language, image GPS metadata, duplicate text, and available
   business history.
6. The backend combines these signals into a heuristic risk score from
   0 to 100.
7. The frontend displays the prediction, risk score, detected signals,
   GPS status, and majority-analysis details.
High-level workflow
User
  |
  v
Streamlit Frontend
  |  review text + rating + image
  v
FastAPI /predict Endpoint
  |
  +--> Input validation and image loading
  |
  +--> TF-IDF + Logistic Regression score
  |
  +--> Rule-based text and rating checks
  |
  +--> Optional image EXIF GPS / distance check
  |
  +--> Duplicate and business-history checks
  |
  v
Combined Heuristic Risk Score
  |
  v
FAKE / GENUINE / NEEDS_REVIEW
  |
  v
Results displayed in Streamlit
Tech Stack
  Component               Technology              Purpose
  Frontend                Streamlit               Interactive user
                                                  interface
  Backend                 FastAPI                 Prediction API
  API server              Uvicorn                 Runs the FastAPI
                                                  application
  Machine learning        scikit-learn            TF-IDF vectorization
                                                  and Logistic Regression
  Sparse features         SciPy                   Combines text and
                                                  numeric features
  Image processing        Pillow                  Opens images and reads
                                                  EXIF metadata
  Text processing         Python re,            Normalization, counts,
                          collections,          and duplicate
                          difflib               comparison
  HTTP client             Requests                Sends frontend inputs
                                              to the backend
Project Structure
A simple suggested structure is:
fake-review-detector/
├── backend/
│   └── main.py          # FastAPI code
├── frontend/
│   └── app.py           # Streamlit code
├── requirements.txt
└── README.md
The filenames and folders are suggestions. If your files are named
differently, use your actual paths in the commands below.
Requirements
- Python 3.10 or a compatible newer Python version
- pip
- An internet connection for installing Python packages
- A modern web browser
Installation
1. Clone or download the project
If the project is hosted on GitHub:
git clone <YOUR_REPOSITORY_URL>
cd fake-review-detector
Alternatively, download the project files and open a terminal in the
project directory.
2. Create and activate a virtual environment
Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
Windows (Command Prompt):
python -m venv .venv
.venv\Scripts\activate.bat
macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
Create a requirements.txt file with:
fastapi
uvicorn[standard]
python-multipart
pillow
scipy
scikit-learn
streamlit
requests
Then install the packages:
pip install -r requirements.txt
python-multipart is required by FastAPI for file uploads and form
fields.
Run the Application
The frontend and backend run as separate processes. Keep both terminals
open while using the application.
Terminal 1: Start the FastAPI backend
From the project root, if your backend file is backend/main.py and the
FastAPI object is named app:
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
If your backend file is named main.py in the root directory instead,
run:
uvicorn main:app --reload --host 127.0.0.1 --port 8000
The API should be available at:
- API base URL: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs
Terminal 2: Start the Streamlit frontend
If your frontend file is frontend/app.py, run:
streamlit run frontend/app.py
If it is named app.py in the root directory, run:
streamlit run app.py
Streamlit will display a local URL in the terminal, usually
http://localhost:8501. Open it in your browser.
Configure the backend URL (optional)
The frontend reads the backend URL from the API_URL environment
variable. By default, it uses http://127.0.0.1:8000.
PowerShell:
$env:API_URL="http://127.0.0.1:8000"
streamlit run frontend/app.py
macOS / Linux:
export API_URL="http://127.0.0.1:8000"
streamlit run frontend/app.py
Change the URL if the backend is hosted elsewhere.
Using the App
1. Open the Streamlit frontend in your browser.
2. Upload a JPG/JPEG or PNG image.
3. Enter the review text.
4. Select a star rating from 1 to 5.
5. Optionally enter the business/institute ID. The sample history is
   associated with the ID demo.
6. Adjust the majority-outlier threshold if needed.
7. Enable or disable the severe-abuse rule.
8. Click Analyze Review.
9. Review the prediction, risk score, detected signals, image GPS
   status, and majority-analysis output.
The current interface also contains optional reviewer-name and
location-name fields. These fields are sent by the frontend but are
not used by the current backend prediction logic.
API Reference
POST /predict
Accepts a multipart form containing an image and review details.
Required fields
  Field                   Type                    Description
  image                 File                    JPG/JPEG or PNG image
  review_text           String                  Review text; must
                                                  contain at least 3
                                                  non-whitespace
                                                  characters
  rating                Number                  Star rating from 1 to 5
Optional fields
  Field                   Type              Default           Description
  business_id           String            None            Key used to look
                                                              up review history
  business_latitude     Number            None            Business latitude
                                                              for image GPS
                                                              distance
                                                              comparison
  business_longitude    Number            None            Business longitude
                                                              for image GPS
                                                              distance
                                                              comparison
  majority_threshold    Number            0.8             Share threshold
                                                              for
                                                              majority-outlier
                                                              analysis
  severe_rule_enabled   Boolean           true            Enables or
                                                              disables the
                                                          severe-abuse rule
The endpoint rejects ratings outside 1--5, review text shorter than 3
characters, content types other than JPEG/PNG, images larger than 5 MB,
and files that Pillow cannot open as images.
Example API request
You can test the endpoint using curl:
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "image=@sample.jpg" \
  -F "review_text=The food was great and the staff were friendly" \
  -F "rating=5" \
  -F "business_id=demo" \
  -F "majority_threshold=0.8" \
  -F "severe_rule_enabled=true"
Replace sample.jpg with the path to a real local image file.
Example response
The exact values depend on the input:
{
  "prediction": "NEEDS_REVIEW",
  "heuristic_ml_score": 0.42,
  "risk_score": 35,
  "detected_signals": [
    "Extreme rating (weak signal)"
  ],
  "image_gps_status": "GPS metadata not available (missing evidence, not proof of fraud)",
  "rating_sentiment_match": true,
  "majority_analysis": {
    "available": true,
    "reviews": 100,
    "positive_share": 0.9,
    "negative_share": 0.1,
    "threshold": 0.8,
    "outlier": false
  },
  "explanation": "NEEDS_REVIEW: risk 35/100. Extreme rating (weak signal)"
}
This is an illustrative response shape, not a guaranteed result for the
example input.
Understanding the Results
Prediction
- GENUINE --- The heuristic risk score is below 35 and no
  overriding rule changes the result.
- NEEDS_REVIEW --- The score is generally between 35 and 59, or
  a majority-outlier signal requires further review.
- FAKE --- The score is generally 60 or higher, or the enabled
  severe-abuse rule triggers the current override.
These labels are outputs of the current demo logic; they are not
verified judgments about a review's authenticity.
Scores
- heuristic_ml_score --- The Logistic Regression model's
  predicted class score for the synthetic training setup. It is not
  calibrated and should not be interpreted as a trustworthy real-world
  probability.
- risk_score --- A rule-and-model-based score from 0 to 100. It
  adds weighted signals and caps the result at 100. Some signals can
  overlap, so the score is not a statistically validated fraud
  probability.
Detected signals
Signals explain which checks were triggered, such as rating--sentiment
mismatch, promotional wording, repeated words, excessive capitalization,
duplicate text, or a majority-outlier pattern. Some signals are weak
indicators and can occur in legitimate reviews.
Image GPS status
GPS coordinates can only be read when they are present in supported
image EXIF metadata. Many platforms remove metadata when images are
uploaded or shared. Missing GPS data is treated as missing evidence, not
proof of fraud.
If business coordinates are not supplied to the API, the system cannot
calculate the distance between the image location and the business. The
current Streamlit frontend does not expose business latitude and
longitude inputs, so this check requires API-level input or a future
frontend enhancement.
Majority analysis
The demo includes a hard-coded synthetic history for business_id=demo.
Other IDs have no stored history unless the code is extended. Majority
analysis needs at least MIN_HISTORY historical reviews (currently 5)
and compares positive ratings (4--5 stars) and negative ratings (1--2
stars). A review that differs from the majority is an outlier, but an
outlier is not necessarily fake.
Current Limitations
- The training dataset is small, manually written, and synthetic; it
  does not represent real-world review behavior.
- The classifier and risk score have not been validated on a
  representative labeled dataset.
- The ML score is not calibrated and should not be presented as a
  probability that a review is fake.
- Word lists and phrase rules can miss subtle manipulation and can
  incorrectly flag legitimate language.
- A rating--sentiment mismatch or an opinion that differs from the
  majority is not proof of fraud.
- Duplicate matching uses a simple string-similarity measure and may
  miss paraphrases or flag common phrases.
- Business history is hard-coded for the demo ID rather than
  retrieved from a database.
- GPS metadata may be absent, stripped, inaccurate, or manipulated.
  Location alone cannot establish review authenticity.
- The frontend's reviewer-name and location-name fields are currently
  unused by the backend.
- The frontend does not currently provide business latitude/longitude
  fields for the GPS distance check.
- The application is intended for local demonstration and has not been
  hardened for production deployment.
Possible Improvements
- Train and evaluate the model on a larger, real, ethically sourced,
  labeled dataset.
- Use a held-out test set and report precision, recall, F1-score,
  confusion matrix, and false-positive rates.
- Calibrate model probabilities and clearly separate model output from
  rule-based risk scoring.
- Store business reviews in a database and implement a reliable
  history retrieval pipeline.
- Add a configurable threshold and explain how each signal affects the
  final score.
- Add business coordinates to the frontend and validate geographic
  inputs.
- Add tests for input validation, prediction rules, duplicate
  detection, and API responses.
- Improve file validation, error handling, logging, rate limiting, and
  privacy protections before any public deployment.
- Avoid making automatic punitive decisions based solely on this
  detector; provide human review and an appeal process.
Disclaimer
This project is intended for learning, demonstration, and
experimentation. It should not be used as the sole basis for removing
reviews, penalizing reviewers, or making business decisions. Review
authenticity requires broader evidence and careful human assessment.

HEART DISEASE RISK ESTIMATOR WEBSITE

FILES
- app.py: Flask website and Logistic Regression model based on your Python script
- templates/index.html: website form and results page
- requirements.txt: Python packages needed
- heart_disease_dataset_10000.csv: YOU must add your own dataset here

IMPORTANT
The CSV file must be in the same folder as app.py and must contain these columns:
age, sex, resting_blood_pressure_mmHg, cholesterol_mg_dL, bmi,
fasting_blood_sugar_high, diabetes, smoking, family_history_heart_disease,
exercise_induced_angina, maximum_heart_rate, chest_pain_type, heart_disease

The target column heart_disease must contain 0 and 1.

HOW TO RUN (Mac / Windows)
1. Install Python.
2. Open Terminal (Mac) or Command Prompt (Windows).
3. Move into this folder:
   cd path/to/heart_disease_website
4. Install packages:
   python -m pip install -r requirements.txt
5. Copy heart_disease_dataset_10000.csv into this folder.
6. Start the website:
   python app.py
7. Open this address in your browser:
   http://127.0.0.1:5000

If the command 'python' does not work on Mac, try 'python3'.

The model trains from the CSV when the app starts. Restart the app after replacing the CSV.

MEDICAL DISCLAIMER
This is an educational model output, not a diagnosis or a clinically validated individual risk score.
Do not use it for medical decisions. Synthetic or unvalidated data can produce misleading results.

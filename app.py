import streamlit as st
import pickle
import warnings
import numpy as np
import pandas as pd # Import pandas for the ideal fix (optional for suppression)

# Import specific warning classes
from sklearn.exceptions import InconsistentVersionWarning
# from warnings import UserWarning # Not strictly needed if filtering by message

# --- Suppress Scikit-learn Warnings ---
# 1. Version Mismatch Warning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
# 2. Feature Names Warning (Suppression method)
warnings.filterwarnings("ignore", message="X does not have valid feature names, but", category=UserWarning)
# ---------------------------------------

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️")

# --- Hide Streamlit Header/Footer ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- Background Image and CSS Styling ---
background_image_url = "https://www.strategyand.pwc.com/m1/en/strategic-foresight/sector-strategies/healthcare/ai-powered-healthcare-solutions/img01-section1.jpg"

page_bg_img = f"""
<style>
/* Background Image */
[data-testid="stAppViewContainer"] {{
    background-image: url("{background_image_url}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* Dark Overlay */
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.75); /* Slightly darker overlay */
}}

/* --- General Text Styling --- */
/* Apply white text more selectively to common elements */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] p, /* General paragraphs */
[data-testid="stAppViewContainer"] .stMarkdown, /* Content from st.write/st.markdown */
[data-testid="stAppViewContainer"] .stCaption, /* Content from st.caption */
[data-testid="stSidebar"] * /* All sidebar content */
{{
    color: white !important; /* Use important to override defaults if needed */
}}

/* Style labels for input elements specifically */
.stTextInput label,
.stNumberInput label,
.stSelectbox label {{
    color: #e0e0e0 !important; /* Lighter gray for labels */
}}

/* --- Input Field Text & Placeholder Styling (FIX) --- */

/* Ensure text TYPED INTO the input is black */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {{
    color: black !important;
}}

/* Make PLACEHOLDER text black and slightly faded */
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stNumberInput"] input::placeholder {{
    color: black !important;
    opacity: 0.6 !important; /* Adjust opacity as needed */
}}
/* Vendor prefixes for placeholder */
div[data-testid="stTextInput"] input::-webkit-input-placeholder,
div[data-testid="stNumberInput"] input::-webkit-input-placeholder {{ color: black !important; opacity: 0.6 !important; }}
div[data-testid="stTextInput"] input:-moz-placeholder,
div[data-testid="stNumberInput"] input:-moz-placeholder {{ color: black !important; opacity: 0.6 !important; }}
div[data-testid="stTextInput"] input::-moz-placeholder,
div[data-testid="stNumberInput"] input::-moz-placeholder {{ color: black !important; opacity: 0.6 !important; }}
div[data-testid="stTextInput"] input:-ms-input-placeholder,
div[data-testid="stNumberInput"] input:-ms-input-placeholder {{ color: black !important; opacity: 0.6 !important; }}
div[data-testid="stTextInput"] input::-ms-input-placeholder,
div[data-testid="stNumberInput"] input::-ms-input-placeholder {{ color: black !important; opacity: 0.6 !important; }}


/* --- Button Styling --- */
.stButton>button {{
    color: #4F8BF9; background-color: white; border: 1px solid #4F8BF9;
    border-radius: 0.25rem; padding: 0.375rem 0.75rem;
}}
.stButton>button:hover {{ color: white; background-color: #4F8BF9; border: 1px solid white; }}
.stButton>button:active {{ background-color: #3a7bd5; border-color: white; }}
.stButton>button:focus {{ outline: none; box-shadow: none; }}


/* --- Notification Styling (Success, Error, Warning) --- */
/* General notification box styling */
[data-testid="stNotification"] {{
    background-color: rgba(255, 255, 255, 0.95) !important;
    color: black !important;
    border-radius: 0.25rem;
    padding: 0.75rem 1.25rem;
    border: 1px solid #ccc !important;
}}
/* Ensure text inside notifications is black */
[data-testid="stNotification"] p,
[data-testid="stNotification"] li {{
    color: black !important;
}}
/* Specific background/border for success */
.stSuccess[data-testid="stNotification"] {{
     background-color: #D4EDDA !important; border-color: #C3E6CB !important;
}}
/* Specific background/border for error */
[data-testid="stNotification"][kind="error"] {{
    background-color: rgba(248, 215, 218, 0.95) !important; border-color: #f5c6cb !important;
}}
/* Specific background/border for warning */
[data-testid="stNotification"][kind="warning"] {{
    background-color: rgba(255, 243, 205, 0.95) !important; border-color: #ffeeba !important;
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# --- Load Models ---
MODEL_DIR = 'Models'
try:
    models = {
        'diabetes': pickle.load(open(f'{MODEL_DIR}/diabetes_model.sav', 'rb')),
        'heart_disease': pickle.load(open(f'{MODEL_DIR}/heart_disease_model.sav', 'rb')),
        'parkinsons': pickle.load(open(f'{MODEL_DIR}/parkinsons_model.sav', 'rb')),
        'lung_cancer': pickle.load(open(f'{MODEL_DIR}/lungs_disease_model.sav', 'rb')), # Check filename
        'thyroid': pickle.load(open(f'{MODEL_DIR}/Thyroid_model.sav', 'rb'))
    }
except FileNotFoundError as e:
    st.error(f"ERROR: Model file not found: {e}. Check '{MODEL_DIR}' path and filenames.")
    st.stop()
except pickle.UnpicklingError as e:
    st.error(f"ERROR: Could not load model file: {e}. File may be corrupt or incompatible.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred loading models: {type(e).__name__} - {e}")
    st.stop()
# -----------------------------

# --- Disease Selection Dropdown ---
st.title("🩺 Disease Prediction System")

selected = st.selectbox(
    'Select a Disease to Predict',
    ['Diabetes Prediction',
     'Heart Disease Prediction',
     'Parkinsons Prediction',
     'Lung Cancer Prediction',
     'Hypo-Thyroid Prediction'],
     key='disease_select',
     placeholder="Choose an option..." # Add placeholder text to selectbox
)

# --- Input Helper Function (No changes from previous version) ---
def display_input(label, tooltip, key, type="number", min_value=None, max_value=None, step=None, format_str=None):
    is_likely_int = False
    int_indicators = ["Age", "Number", "Sex", "Smoking", "Yellow Fingers", "Anxiety", "Peer Pressure", "Chronic Disease", "Fatigue", "Allergy", "Wheezing", "Alcohol Consuming", "Coughing", "Shortness Of Breath", "Swallowing Difficulty", "Chest Pain", "On Thyroxine", "Query On Thyroxine", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "Gender"]
    if any(indicator in label for indicator in int_indicators) or "1 =" in tooltip or "0 =" in tooltip:
        is_likely_int = True
        if any(float_indicator in label for float_indicator in ["BMI", "Pedigree", "oldpeak", "Hz", "(%)", "(Abs)", "MDVP", "Jitter", "Shimmer", "NHR", "HNR", "RPDE", "DFA", "spread", "D2", "PPE"]):
            is_likely_int = False
    resolved_type = int if is_likely_int else float
    if min_value is None: min_value = 0 if resolved_type == int else 0.0
    else: min_value = resolved_type(min_value)
    if step is None:
        if resolved_type == int: step = 1
        elif "BMI" in label or "Pedigree" in label or "oldpeak" in label: step = 0.1
        elif "(Abs)" in label: step = 0.00001
        elif any(indicator in label for indicator in ["Hz", "(%)", "MDVP", "Jitter", "Shimmer", "NHR", "HNR", "RPDE", "DFA", "spread", "D2", "PPE"]): step = 0.001
        else: step = 0.1
    else: step = resolved_type(step)
    if max_value is not None: max_value = resolved_type(max_value)
    elif resolved_type == int and max_value is None and ("Yes; 0 = No" in tooltip or "male; 0 = female" in tooltip): max_value = 1
    if format_str is None:
        if resolved_type == int: format_str = "%d"
        else:
            step_str = str(step);
            if '.' in step_str: decimals = len(step_str.split('.')[-1]); format_str = f"%.{decimals}f"
            else: format_str = "%.1f"
    if type == "text": return st.text_input(label, key=key, help=tooltip)
    elif type == "number": return st.number_input(label=label, key=key, help=tooltip, min_value=min_value, max_value=max_value, step=step, format=format_str)

# --- Input Sections & Prediction Logic (No major changes needed in logic) ---

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.header('Diabetes Prediction')
    st.write("Enter the following details:") # This text will be white due to CSS above
    n_features_expected = 8
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = display_input('Number of Pregnancies', 'Enter number of times pregnant', 'Pregnancies')
        Glucose = display_input('Glucose Level', 'Enter plasma glucose concentration (mg/dL)', 'Glucose')
        BloodPressure = display_input('Blood Pressure (mm Hg)', 'Enter diastolic blood pressure', 'BloodPressure')
    with col2:
        SkinThickness = display_input('Skin Thickness (mm)', 'Enter triceps skin fold thickness', 'SkinThickness')
        Insulin = display_input('Insulin Level (mu U/ml)', 'Enter 2-Hour serum insulin', 'Insulin')
        BMI = display_input('BMI (kg/m²)', 'Enter Body Mass Index value', 'BMI')
    with col3:
        DiabetesPedigreeFunction = display_input('Diabetes Pedigree Function', 'Enter diabetes pedigree function value', 'DiabetesPedigreeFunction')
        Age = display_input('Age (years)', 'Enter age of the person', 'Age')

    if st.button('Predict Diabetes', key='diabetes_predict'):
        try:
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            if len(input_data) != n_features_expected: st.error(f"Input Error: Expected {n_features_expected} features, got {len(input_data)}.")
            else:
                input_data_np = np.asarray([float(x) for x in input_data]).reshape(1, -1)
                diab_prediction = models['diabetes'].predict(input_data_np)
                diab_diagnosis = '**Result:** The person is **Diabetic**' if diab_prediction[0] == 1 else '**Result:** The person is **Not Diabetic**'
                st.success(diab_diagnosis) # Success box style handled by CSS
        except ValueError as ve: st.error(f"Input Error: Enter valid numbers only. Details: {ve}")
        except Exception as e: st.error(f"Prediction Error: {type(e).__name__} - {e}")

# Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    st.header('Heart Disease Prediction')
    st.write("Enter the following details:")
    n_features_expected = 13
    col1, col2, col3 = st.columns(3)
    with col1:
        age = display_input('Age (years)', 'Enter age', 'age_hd')
        sex = display_input('Sex', '1 = male; 0 = female', 'sex_hd')
        cp = display_input('Chest Pain Type (cp)', '0-3: typical, atypical, non-anginal, asymptomatic', 'cp_hd')
        trestbps = display_input('Resting Blood Pressure (mm Hg)', 'Enter resting blood pressure', 'trestbps_hd')
        chol = display_input('Serum Cholesterol (mg/dl)', 'Enter serum cholesterol', 'chol_hd')
    with col2:
        fbs = display_input('Fasting Blood Sugar > 120 mg/dl (fbs)', '1 = true; 0 = false', 'fbs_hd')
        restecg = display_input('Resting ECG Results (restecg)', '0, 1, or 2', 'restecg_hd')
        thalach = display_input('Max Heart Rate Achieved (thalach)', 'Enter maximum heart rate', 'thalach_hd')
        exang = display_input('Exercise Induced Angina (exang)', '1 = yes; 0 = no', 'exang_hd')
    with col3:
        oldpeak = display_input('ST Depression (oldpeak)', 'ST depression induced by exercise relative to rest', 'oldpeak_hd')
        slope = display_input('Slope of Peak Exercise ST Segment', '0, 1, or 2', 'slope_hd')
        ca = display_input('Major Vessels Colored by Fluoroscopy (ca)', 'Number of major vessels (0-3)', 'ca_hd')
        thal = display_input('Thalassemia (thal)', '0=normal; 1=fixed defect; 2=reversible defect (check dataset codes: 0/1/2 or 1/2/3)', 'thal_hd')

    if st.button('Predict Heart Disease', key='heart_predict'):
        try:
            input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            if len(input_data) != n_features_expected: st.error(f"Input Error: Expected {n_features_expected} features, got {len(input_data)}.")
            else:
                input_data_np = np.asarray([float(x) for x in input_data]).reshape(1, -1)
                heart_prediction = models['heart_disease'].predict(input_data_np)
                heart_diagnosis = '**Result:** The person **Has Heart Disease**' if heart_prediction[0] == 1 else '**Result:** The person **Does Not Have Heart Disease**'
                st.success(heart_diagnosis)
        except ValueError as ve: st.error(f"Input Error: Enter valid numbers only. Details: {ve}")
        except Exception as e: st.error(f"Prediction Error: {type(e).__name__} - {e}")

# Parkinson's Prediction Page
elif selected == "Parkinsons Prediction":
    st.header("Parkinson's Disease Prediction")
    st.write("Enter the following vocal measurement details:")
    n_features_expected = 22
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fo = display_input('MDVP:Fo(Hz)', 'Average vocal fundamental frequency', 'fo_pd')
        fhi = display_input('MDVP:Fhi(Hz)', 'Maximum vocal fundamental frequency', 'fhi_pd')
        flo = display_input('MDVP:Flo(Hz)', 'Minimum vocal fundamental frequency', 'flo_pd')
        Jitter_percent = display_input('MDVP:Jitter(%)', 'MDVP jitter in percentage', 'Jitter_percent_pd')
        Jitter_Abs = display_input('MDVP:Jitter(Abs)', 'MDVP absolute jitter in ms', 'Jitter_Abs_pd')
        RAP = display_input('MDVP:RAP', 'MDVP Relative Amplitude Perturbation', 'RAP_pd')
    with col2:
        PPQ = display_input('MDVP:PPQ', 'MDVP five-point Period Perturbation Quotient', 'PPQ_pd')
        DDP = display_input('Jitter:DDP', 'Average absolute difference of differences between jitter cycles', 'DDP_pd')
        Shimmer = display_input('MDVP:Shimmer', 'MDVP local shimmer', 'Shimmer_pd')
        Shimmer_dB = display_input('MDVP:Shimmer(dB)', 'MDVP local shimmer in dB', 'Shimmer_dB_pd')
        APQ3 = display_input('Shimmer:APQ3', 'Three-point Amplitude Perturbation Quotient', 'APQ3_pd')
        APQ5 = display_input('Shimmer:APQ5', 'Five-point Amplitude Perturbation Quotient', 'APQ5_pd')
    with col3:
        APQ = display_input('MDVP:APQ', 'MDVP 11-point Amplitude Perturbation Quotient', 'APQ_pd')
        DDA = display_input('Shimmer:DDA', 'Average absolute difference between consecutive differences in amplitude', 'DDA_pd')
        NHR = display_input('NHR', 'Noise-to-Harmonics Ratio', 'NHR_pd')
        HNR = display_input('HNR', 'Harmonics-to-Noise Ratio', 'HNR_pd')
        RPDE = display_input('RPDE', 'Recurrence Period Density Entropy measure', 'RPDE_pd')
        DFA = display_input('DFA', 'Signal fractal scaling exponent', 'DFA_pd')
    with col4:
        spread1 = display_input('spread1', 'Nonlinear fundamental frequency variation measure 1', 'spread1_pd')
        spread2 = display_input('spread2', 'Nonlinear fundamental frequency variation measure 2', 'spread2_pd')
        D2 = display_input('D2', 'Correlation dimension', 'D2_pd')
        PPE = display_input('PPE', 'Pitch Period Entropy', 'PPE_pd')

    if st.button("Predict Parkinson's", key='parkinsons_predict'):
        try:
            input_data = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            if len(input_data) != n_features_expected: st.error(f"Input Error: Expected {n_features_expected} features, got {len(input_data)}.")
            else:
                input_data_np = np.asarray([float(x) for x in input_data]).reshape(1, -1)
                parkinsons_prediction = models['parkinsons'].predict(input_data_np)
                parkinsons_diagnosis = "**Result:** The person **Has Parkinson's Disease**" if parkinsons_prediction[0] == 1 else "**Result:** The person **Does Not Have Parkinson's Disease**"
                st.success(parkinsons_diagnosis)
        except ValueError as ve: st.error(f"Input Error: Enter valid numbers only. Details: {ve}")
        except Exception as e: st.error(f"Prediction Error: {type(e).__name__} - {e}")

# Lung Cancer Prediction Page
elif selected == "Lung Cancer Prediction":
    st.header("Lung Cancer Prediction")
    st.write("Enter the following details based on symptoms and habits:")
    n_features_expected = 15
    col1, col2, col3 = st.columns(3)
    with col1:
        GENDER = display_input('Gender', '1 = Male; 0 = Female', 'GENDER_lc')
        AGE = display_input('Age (years)', 'Enter age', 'AGE_lc')
        SMOKING = display_input('Smoking', 'Enter 1 for No, 2 for Yes (Verify!)', 'SMOKING_lc')
        YELLOW_FINGERS = display_input('Yellow Fingers', 'Enter 1 for No, 2 for Yes (Verify!)', 'YELLOW_FINGERS_lc')
        ANXIETY = display_input('Anxiety', 'Enter 1 for No, 2 for Yes (Verify!)', 'ANXIETY_lc')
    with col2:
        PEER_PRESSURE = display_input('Peer Pressure', 'Enter 1 for No, 2 for Yes (Verify!)', 'PEER_PRESSURE_lc')
        CHRONIC_DISEASE = display_input('Chronic Disease', 'Enter 1 for No, 2 for Yes (Verify!)', 'CHRONIC_DISEASE_lc')
        FATIGUE = display_input('Fatigue', 'Enter 1 for No, 2 for Yes (Verify!)', 'FATIGUE_lc')
        ALLERGY = display_input('Allergy', 'Enter 1 for No, 2 for Yes (Verify!)', 'ALLERGY_lc')
        WHEEZING = display_input('Wheezing', 'Enter 1 for No, 2 for Yes (Verify!)', 'WHEEZING_lc')
    with col3:
        ALCOHOL_CONSUMING = display_input('Alcohol Consuming', 'Enter 1 for No, 2 for Yes (Verify!)', 'ALCOHOL_CONSUMING_lc')
        COUGHING = display_input('Coughing', 'Enter 1 for No, 2 for Yes (Verify!)', 'COUGHING_lc')
        SHORTNESS_OF_BREATH = display_input('Shortness Of Breath', 'Enter 1 for No, 2 for Yes (Verify!)', 'SHORTNESS_OF_BREATH_lc')
        SWALLOWING_DIFFICULTY = display_input('Swallowing Difficulty', 'Enter 1 for No, 2 for Yes (Verify!)', 'SWALLOWING_DIFFICULTY_lc')
        CHEST_PAIN = display_input('Chest Pain', 'Enter 1 for No, 2 for Yes (Verify!)', 'CHEST_PAIN_lc')

    st.warning("Note: Verify the numerical coding (e.g., 1/0 or 2/1) for Yes/No inputs required by the Lung Cancer model.")

    if st.button("Predict Lung Cancer", key='lung_cancer_predict'):
        try:
            input_data = [GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]
            if len(input_data) != n_features_expected: st.error(f"Input Error: Expected {n_features_expected} features, got {len(input_data)}.")
            else:
                input_data_np = np.asarray([float(x) for x in input_data]).reshape(1, -1)
                lungs_prediction = models['lung_cancer'].predict(input_data_np)
                prediction_val = lungs_prediction[0]
                if prediction_val == 1 or str(prediction_val).upper() == 'YES' or prediction_val == 2: lungs_diagnosis = "**Result:** The person likely **Has Lung Cancer**"
                else: lungs_diagnosis = "**Result:** The person likely **Does Not Have Lung Cancer**"
                st.success(lungs_diagnosis)
        except ValueError as ve: st.error(f"Input Error: Enter valid numbers only (using correct Yes/No coding). Details: {ve}")
        except Exception as e: st.error(f"Prediction Error: {type(e).__name__} - {e}")

# Hypo-Thyroid Prediction Page
elif selected == "Hypo-Thyroid Prediction":
    st.header("Hypo-Thyroid Prediction")
    st.write("Enter the following details:")
    n_features_expected = 4 # !!! PLACEHOLDER - UPDATE THIS !!!
    st.error(f"CRITICAL WARNING: This section currently assumes the Thyroid model ONLY needs {n_features_expected} features. This is highly unlikely. Update the code to collect all features the specific `Thyroid_model.sav` requires.")

    col1, col2 = st.columns(2)
    with col1:
        age_t = display_input('Age (years)', 'Enter age', 'age_thyroid')
        sex_t = display_input('Sex', '1 = Male; 0 = Female', 'sex_thyroid')
    with col2:
        on_thyroxine = display_input('On Thyroxine Medication', '1 = Yes; 0 = No', 'on_thyroxine_thyroid')
        query_on_thyroxine = display_input('Query On Thyroxine', '1 = Yes; 0 = No', 'query_on_thyroxine_thyroid')

    # --- !!! ADD MORE INPUTS HERE AS NEEDED FOR YOUR ACTUAL THYROID MODEL !!! ---

    if st.button("Predict Hypothyroidism", key='thyroid_predict'):
        try:
            input_data = [age_t, sex_t, on_thyroxine, query_on_thyroxine] # Add ALL required features
            if len(input_data) != n_features_expected: st.error(f"Input Error: Expected {n_features_expected} features (update code!), got {len(input_data)}.")
            else:
                input_data_np = np.asarray([float(x) for x in input_data]).reshape(1, -1)
                thyroid_prediction = models['thyroid'].predict(input_data_np)
                thyroid_diagnosis = '**Result:** The person likely **Has Hypothyroidism**' if thyroid_prediction[0] == 1 else '**Result:** The person likely **Does Not Have Hypothyroidism**'
                st.success(thyroid_diagnosis)
        except ValueError as ve: st.error(f"Input Error: Enter valid numbers only. Details: {ve}")
        except Exception as e: st.error(f"Prediction Error: {type(e).__name__} - {e}. Verify required features for Thyroid model.")

# --- Footer ---
st.markdown("---") # This markdown's text color should be white
st.caption("Disclaimer: This tool provides predictions based on statistical models and should not replace professional medical advice.") # Caption color should be white
import os
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Hub",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'{working_dir}/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/parkinsons_model.sav', 'rb'))

autism_prediction=pickle.load(open(f'{working_dir}/best_model.sav', 'rb'))

breastcancer_model=pickle.load(open(f'{working_dir}/breastcancer_model.sav', 'rb'))

alzheimers_model=pickle.load(open(f'{working_dir}/alzheimers_model.sav', 'rb'))

lungcancer_model=pickle.load(open(f'{working_dir}/Lung Cancer_model.sav', 'rb'))

typhoid_model=pickle.load(open(f'{working_dir}/typhoid_model.sav', 'rb'))

brainstroke_model=pickle.load(open(f'{working_dir}/brain stroke_model.sav', 'rb'))

thyroid_model=pickle.load(open(f'{working_dir}/thyroid_model.sav', 'rb'))


# sidebar for navigation
with st.sidebar:
    selected = option_menu('Gateway to your health hub',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Autism Prediction',
                            'Breast Cancer Survival',
                            'Alzheimers Check',
                            'Lung Cancer Prediction',
                            'Typhoid Test',
                            'Brain Stroke',
                            'Thyroid Test'],
                           
                           icons=['activity', 'heart', 'person','activity','plus','activity','lungs','plus','person','heart'],
                           menu_icon='hospital-fill',
                           
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # Page title
    st.title('Diabetes Prediction')

    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)

    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0.0, step=0.1)

    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0.0, step=0.1)

    with col1:
        SkinThickness = st.number_input('Skin Thickness value', min_value=0.0, step=0.1)

    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0.0, step=0.1)

    with col3:
        BMI = st.number_input('BMI value', min_value=0.0, step=0.1)

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, step=0.01)

    with col2:
        Age = st.number_input('Age of the Person', min_value=0, step=1)

    # Code for Prediction
    diab_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            # Collecting user inputs
            user_input = [
                Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                BMI, DiabetesPedigreeFunction, Age
            ]

            # Model Prediction
            diab_prediction = diabetes_model.predict([user_input])

            # Checking prediction result
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'

            # Displaying result
            st.success(diab_diagnosis)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # Page title
    st.title('Heart Disease Prediction')

    # Add Mappings for Categorical Features
    sex_map = {'Male': 0, 'Female': 1}
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
    fbs_map = {'True': 1, 'False': 0}
    restecg_map = {'Normal': 0, 'Having ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    exang_map = {'Yes': 1, 'No': 0}
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    ca_map = {'0': 0, '1': 1, '2': 2, '3': 3}
    thal_map = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}

    # Collecting user inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0)

    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'])

    with col3:
        cp = st.selectbox('Chest Pain types', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])

    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=0)

    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0)

    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])

    with col1:
        restecg = st.selectbox('Resting Electrocardiographic results', ['Normal', 'Having ST-T wave abnormality', 'Left ventricular hypertrophy'])

    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=0)

    with col3:
        exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])

    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0)

    with col2:
        slope = st.selectbox('Slope of the peak exercise ST segment', ['Upsloping', 'Flat', 'Downsloping'])

    with col3:
        ca = st.selectbox('Major vessels colored by fluoroscopy', ['0', '1', '2', '3'])

    with col1:
        thal = st.selectbox('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    # Code for Prediction
    heart_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Heart Disease Test Result'):

        try:
            # Mapping Categorical Data
            sex_encoded = sex_map[sex]
            cp_encoded = cp_map[cp]
            fbs_encoded = fbs_map[fbs]
            restecg_encoded = restecg_map[restecg]
            exang_encoded = exang_map[exang]
            slope_encoded = slope_map[slope]
            ca_encoded = ca_map[ca]
            thal_encoded = thal_map[thal]

            # Collecting and encoding the user inputs
            user_input = [
                age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded, restecg_encoded, 
                thalach, exang_encoded, oldpeak, slope_encoded, ca_encoded, thal_encoded
            ]

            # Model Prediction
            heart_prediction = heart_disease_model.predict([user_input])

            # Checking prediction result
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'

            # Displaying result
            st.success(heart_diagnosis)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # Page title
    st.title("Parkinson's Disease Prediction ")

    # Collecting user inputs
    col1, col2, col3, col4, col5 = st.columns(5)

    # Collect all input data
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')  # float
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')  # float
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')  # float
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')  # float
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')  # float

    with col1:
        RAP = st.text_input('MDVP:RAP')  # float
    with col2:
        PPQ = st.text_input('MDVP:PPQ')  # float
    with col3:
        DDP = st.text_input('Jitter:DDP')  # float
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')  # float
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')  # float

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')  # float
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')  # float
    with col3:
        APQ = st.text_input('MDVP:APQ')  # float
    with col4:
        DDA = st.text_input('Shimmer:DDA')  # float
    with col5:
        NHR = st.text_input('NHR')  # float

    with col1:
        HNR = st.text_input('HNR')  # float
    with col2:
        RPDE = st.text_input('RPDE')  # float
    with col3:
        DFA = st.text_input('DFA')  # float
    with col4:
        spread1 = st.text_input('spread1')  # float
    with col5:
        spread2 = st.text_input('spread2')  # float

    with col1:
        D2 = st.text_input('D2')  # float
    with col2:
        PPE = st.text_input('PPE')  # float

    # Code for Prediction
    parkinsons_diagnosis = ''

    # Creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        try:
            # Mapping the user inputs into a list of floats
            user_input = [
                fo, fhi, flo, Jitter_percent, Jitter_Abs,
                RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE
            ]

            # Convert all string inputs to float (input validation can be done here as well)
            user_input = [float(x) if x != '' else 0.0 for x in user_input]  # Default 0.0 for empty inputs

            # Prediction using the Parkinson's Disease model
            parkinsons_prediction = parkinsons_model.predict([user_input])

            # Check the result of prediction
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"

            # Displaying the prediction result
            st.success(parkinsons_diagnosis)

        except ValueError as e:
            st.error(f"Error in input conversion: {e}")

    
    # autism prediction
    
# Mapping for categorical inputs (assuming mappings are available)
ethnicity_mapping = {'Ethnicity1': 1, 'Ethnicity2': 2}  # Replace with actual mappings
country_mapping = {'Country1': 1, 'Country2': 2}  # Replace with actual mappings
relation_mapping = {'Relation1': 1, 'Relation2': 2}  # Replace with actual mappings

if selected == 'Autism Prediction':
    # Page title
    st.title('Autism Prediction')

    col1, col2, col3 = st.columns(3)

    # Collecting inputs from the user
    with col1:
        A1_Score = st.number_input("A1 Score (0 or 1)", min_value=0, max_value=1, step=1)

    with col2:
        A2_Score = st.number_input("A2 Score (0 or 1)", min_value=0, max_value=1, step=1)

    with col3:
        A3_Score = st.number_input("A3 Score (0 or 1)", min_value=0, max_value=1, step=1)

    with col1:
        A4_Score = st.number_input("A4 Score (0 or 1)", min_value=0, max_value=1, step=1)

    with col2:
        A5_Score = st.number_input("A5 Score (0 or 1)", min_value=0, max_value=1, step=1)

    with col3:
        A6_Score = st.number_input("A6 Score (0 or 1)", min_value=0, max_value=1, step=1)

    with col1:
        A7_Score = st.number_input("A7 Score (0 or 1)", min_value=0, max_value=1, step=1)

    with col2:
        A8_Score = st.number_input("A8 Score (0 or 1)", min_value=0, max_value=1, step=1)

    with col3:
        A9_Score = st.number_input("A9 Score (0 or 1)", min_value=0, max_value=1, step=1)

    with col1:
        A10_Score = st.number_input("A10 Score (0 or 1)", min_value=0, max_value=1, step=1)

    with col2:
        gender = st.selectbox('Gender', ['Male', 'Female'])

    with col3:
        age = st.number_input('Age', min_value=0)

    with col1:
        ethnicity = st.selectbox('Ethnicity', list(ethnicity_mapping.keys()))

    with col2:
        jaundice = st.selectbox('Jaundice at birth', ['Yes', 'No'])

    with col3:
        autism_family = st.selectbox('Autism in family', ['Yes', 'No'])

    with col1:
        country_of_res = st.selectbox('Country of Residence', list(country_mapping.keys()))

    with col2:
        used_app_before = st.selectbox('Used app before', ['Yes', 'No'])

    with col3:
        result = st.number_input('AQ1-10 Screening Test Score', min_value=0.0)

    with col1:
        relation = st.selectbox('Relation to patient', list(relation_mapping.keys()))

    # Code for Prediction
    autism_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Autism Prediction Test Result'):
        # Mapping categorical inputs to numeric values
        user_input = [
            A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score, A10_Score,
            gender == 'Male',  # Male is 1, Female is 0
            age,
            ethnicity_mapping[ethnicity],
            jaundice == 'Yes',  # Yes is 1, No is 0
            autism_family == 'Yes',  # Yes is 1, No is 0
            country_mapping[country_of_res],
            used_app_before == 'Yes',  # Yes is 1, No is 0
            result,
            relation_mapping[relation]
        ]

        # Predicting using the autism prediction model
        autism_prediction = autism_prediction.predict([user_input])

        # Show the prediction result
        if autism_prediction[0] == 1:
            autism_diagnosis = 'The person is likely to have Autism'
        else:
            autism_diagnosis = 'The person is not likely to have Autism'

    st.success(autism_diagnosis)
    
    #breast cancer survival
    

# Breast Cancer Survival Prediction Page
if selected == 'Breast Cancer Survival':
    st.title('Breast Cancer Survival Prediction')

    # Collecting user inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0)
    with col2:
        race = st.selectbox('Race', ['Caucasian', 'African American', 'Asian', 'Other'])
    with col3:
        marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Widowed'])

    with col1:
        t_stage = st.selectbox('T Stage', ['T1', 'T2', 'T3', 'T4'])
    with col2:
        n_stage = st.selectbox('N Stage', ['N0', 'N1', 'N2', 'N3'])
    with col3:
        sixth_stage = st.selectbox('6th Stage', ['I', 'II', 'III', 'IV'])

    with col1:
        differentiate = st.selectbox('Differentiation', ['Well', 'Moderate', 'Poor'])
    with col2:
        grade = st.selectbox('Grade', ['Low', 'Medium', 'High'])
    with col3:
        a_stage = st.selectbox('A Stage', ['A', 'B', 'C', 'D'])

    with col1:
        tumor_size = st.number_input('Tumor Size (in cm)', min_value=0.0, format="%.1f")
    with col2:
        estrogen_status = st.selectbox('Estrogen Status', ['Positive', 'Negative'])
    with col3:
        progesterone_status = st.selectbox('Progesterone Status', ['Positive', 'Negative'])

    with col1:
        regional_node_examined = st.number_input('Regional Nodes Examined', min_value=0)
    with col2:
        regional_node_positive = st.number_input('Regional Nodes Positive', min_value=0)
    with col3:
        survival_months = st.number_input('Survival Months', min_value=0)
        
    # Mapping categorical inputs
    race_map = {'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Other': 3}
    marital_status_map = {'Single': 0, 'Married': 1, 'Divorced': 2, 'Widowed': 3}
    t_stage_map = {'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3}
    n_stage_map = {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3}
    sixth_stage_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
    differentiate_map = {'Well': 0, 'Moderate': 1, 'Poor': 2}
    grade_map = {'Low': 0, 'Medium': 1, 'High': 2}
    a_stage_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    estrogen_status_map = {'Positive': 1, 'Negative': 0}
    progesterone_status_map = {'Positive': 1, 'Negative': 0}

    # Processing the user input
    user_input = [
        age,
        race_map[race],
        marital_status_map[marital_status],
        t_stage_map[t_stage],
        n_stage_map[n_stage],
        sixth_stage_map[sixth_stage],
        differentiate_map[differentiate],
        grade_map[grade],
        a_stage_map[a_stage],
        tumor_size,
        estrogen_status_map[estrogen_status],
        progesterone_status_map[progesterone_status],
        regional_node_examined,
        regional_node_positive,
        survival_months
    ]

    # Prediction button
    if st.button('Predict Survival'):
        # Process user input to match model requirements
        

        breast_diagnosis = ''
        breastcancer= breastcancer_model.predict([user_input])

        # Show the prediction result
        if breastcancer[0] == 1:
            breast_diagnosis = 'The person does not have a chance of survival'
        else:
            breast_diagnosis = 'The person do have a chance of survival'

        st.success(breast_diagnosis)
    
    
#alzheimers prediction


# Cognitive Decline Prediction Page
if selected == 'Alzheimers Check':
    st.title('Cognitive Decline Prediction ')

      


    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    ethnicity_map = {'Caucasian': 0, 'African American': 1, 'Hispanic': 2, 'Asian': 3, 'Other': 4}
    education_level_map = {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'Doctorate': 3}
    smoking_map = {'Never smoked': 0, 'Formerly smoked': 1, 'Currently smoking': 2}
    alcohol_consumption_map = {'None': 0, 'Light': 1, 'Moderate': 2, 'Heavy': 3}
    physical_activity_map = {'None': 0, 'Light': 1, 'Moderate': 2, 'High': 3}
    diet_quality_map = {'Poor': 0, 'Average': 1, 'Good': 2}
    sleep_quality_map = {'Poor': 0, 'Average': 1, 'Good': 2}
    family_history_map = {'No': 0, 'Yes': 1}
    cardiovascular_disease_map = {'No': 0, 'Yes': 1}
    diabetes_map = {'No': 0, 'Yes': 1}
    depression_map = {'No': 0, 'Yes': 1}
    head_injury_map = {'No': 0, 'Yes': 1}
    hypertension_map = {'No': 0, 'Yes': 1}
    functional_assessment_map = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
    memory_complaints_map = {'No': 0, 'Yes': 1}
    behavioral_problems_map = {'No': 0, 'Yes': 1}
    adl_map = {'Normal': 0, 'Impaired': 1}
    confusion_map = {'No': 0, 'Yes': 1}
    disorientation_map = {'No': 0, 'Yes': 1}
    personality_changes_map = {'No': 0, 'Yes': 1}
    difficulty_completing_tasks_map = {'No': 0, 'Yes': 1}
    forgetfulness_map = {'No': 0, 'Yes': 1}
    thyroid_history_map = {'No': 0, 'Yes': 1}

# Step 2: Create the input form for the user to fill out

# Title of the page


# Collecting user inputs in columns
    col1, col2, col3 = st.columns(3)

    with col1:
     age = st.number_input('Age', min_value=0, max_value=120)
    with col2:
     gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    with col3:
     ethnicity = st.selectbox('Ethnicity', ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'])

    with col1:
     education_level = st.selectbox('Education Level', ['High School', 'Bachelors', 'Masters', 'Doctorate'])
    with col2:
      bmi = st.number_input('BMI', min_value=0.0, max_value=60.0, format="%.1f")
    with col3:
     smoking = st.selectbox('Smoking', ['Never smoked', 'Formerly smoked', 'Currently smoking'])

    with col1:
     alcohol_consumption = st.selectbox('Alcohol Consumption', ['None', 'Light', 'Moderate', 'Heavy'])
    with col2:
     physical_activity = st.selectbox('Physical Activity', ['None', 'Light', 'Moderate', 'High'])
    with col3:
     diet_quality = st.selectbox('Diet Quality', ['Poor', 'Average', 'Good'])

    with col1:
     sleep_quality = st.selectbox('Sleep Quality', ['Poor', 'Average', 'Good'])
    with col2:
     family_history = st.selectbox('Family History of Alzheimer\'s', ['No', 'Yes'])
    with col3:
     cardiovascular_disease = st.selectbox('Cardiovascular Disease', ['No', 'Yes'])

    with col1:
     diabetes = st.selectbox('Diabetes', ['No', 'Yes'])
    with col2:
     depression = st.selectbox('Depression', ['No', 'Yes'])
    with col3:
     head_injury = st.selectbox('Head Injury', ['No', 'Yes'])

    with col1:
     hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    with col2:
     systolic_bp = st.number_input('Systolic Blood Pressure', min_value=50, max_value=200)
    with col3:
     diastolic_bp = st.number_input('Diastolic Blood Pressure', min_value=30, max_value=130)

    with col1:
     cholesterol_total = st.number_input('Total Cholesterol (mg/dL)', min_value=100, max_value=400)
    with col2:
     cholesterol_ldl = st.number_input('LDL Cholesterol (mg/dL)', min_value=50, max_value=200)
    with col3:
     cholesterol_hdl = st.number_input('HDL Cholesterol (mg/dL)', min_value=20, max_value=100)

    with col1:
     cholesterol_triglycerides = st.number_input('Triglycerides (mg/dL)', min_value=50, max_value=500)
    with col2:
     mmse = st.number_input('Mini-Mental State Examination (MMSE)', min_value=0, max_value=30)
    with col3:
     functional_assessment = st.selectbox('Functional Assessment', ['Normal', 'Mild', 'Moderate', 'Severe'])

    with col1:
     memory_complaints = st.selectbox('Memory Complaints', ['No', 'Yes'])
    with col2:
     behavioral_problems = st.selectbox('Behavioral Problems', ['No', 'Yes'])
    with col3:
     adl = st.selectbox('Activities of Daily Living (ADL)', ['Normal', 'Impaired'])

    with col1:
     confusion = st.selectbox('Confusion', ['No', 'Yes'])
    with col2:
     disorientation = st.selectbox('Disorientation', ['No', 'Yes'])
    with col3:
     personality_changes = st.selectbox('Personality Changes', ['No', 'Yes'])

    with col1:
     difficulty_completing_tasks = st.selectbox('Difficulty Completing Tasks', ['No', 'Yes'])
    with col2:
     forgetfulness = st.selectbox('Forgetfulness', ['No', 'Yes'])
    with col3:
     thyroid_history = st.selectbox('Thyroid History', ['No', 'Yes'])

     

# Step 3: Process user input and map to numeric values

    user_input = [
    age,
    gender_map[gender],
    ethnicity_map[ethnicity],
    education_level_map[education_level],
    bmi,
    smoking_map[smoking],
    alcohol_consumption_map[alcohol_consumption],
    physical_activity_map[physical_activity],
    diet_quality_map[diet_quality],
    sleep_quality_map[sleep_quality],
    family_history_map[family_history],
    cardiovascular_disease_map[cardiovascular_disease],
    diabetes_map[diabetes],
    depression_map[depression],
    head_injury_map[head_injury],
    hypertension_map[hypertension],
    systolic_bp,
    diastolic_bp,
    cholesterol_total,
    cholesterol_ldl,
    cholesterol_hdl,
    cholesterol_triglycerides,
    mmse,
    functional_assessment_map[functional_assessment],
    memory_complaints_map[memory_complaints],
    behavioral_problems_map[behavioral_problems],
    adl_map[adl],
    confusion_map[confusion],
    disorientation_map[disorientation],
    personality_changes_map[personality_changes],
    difficulty_completing_tasks_map[difficulty_completing_tasks],
    forgetfulness_map[forgetfulness],
    thyroid_history_map[thyroid_history]
]


        
    alzheimers_prediction = alzheimers_model.predict([user_input])  
     
    if alzheimers_prediction[0] == 1:
         st.success('The person is at risk of Alzheimer\'s disease.')
    else:
         st.success('The person is not at risk of Alzheimer\'s disease.')

    
    #lung cancer prediction
    

# Lung Disease Prediction Page
if selected == 'Lung Cancer Prediction':
    st.title('Lung Disease Prediction')

    # Collecting user inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    with col2:
        age = st.number_input('Age', min_value=0)
    with col3:
        smoking = st.selectbox('Smoking', ['Yes', 'No'])

    with col1:
        yellow_fingers = st.selectbox('Yellow Fingers', ['Yes', 'No'])
    with col2:
        anxiety = st.selectbox('Anxiety', ['Yes', 'No'])
    with col3:
        peer_pressure = st.selectbox('Peer Pressure', ['Yes', 'No'])

    with col1:
        chronic_disease = st.selectbox('Chronic Disease', ['Yes', 'No'])
    with col2:
        fatigue = st.selectbox('Fatigue', ['Yes', 'No'])
    with col3:
        allergy = st.selectbox('Allergy', ['Yes', 'No'])

    with col1:
        wheezing = st.selectbox('Wheezing', ['Yes', 'No'])
    with col2:
        alcohol_consuming = st.selectbox('Alcohol Consuming', ['Yes', 'No'])
    with col3:
        coughing = st.selectbox('Coughing', ['Yes', 'No'])

    with col1:
        shortness_of_breath = st.selectbox('Shortness of Breath', ['Yes', 'No'])
    with col2:
        swallowing_difficulty = st.selectbox('Swallowing Difficulty', ['Yes', 'No'])
    with col3:
        chest_pain = st.selectbox('Chest Pain', ['Yes', 'No'])
        
    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    yes_no_map = {'Yes': 1, 'No': 0}

    # Process user inputs
    user_input = [
        gender_map[gender],
        age,
        yes_no_map[smoking],
        yes_no_map[yellow_fingers],
        yes_no_map[anxiety],
        yes_no_map[peer_pressure],
        yes_no_map[chronic_disease],
        yes_no_map[fatigue],
        yes_no_map[allergy],
        yes_no_map[wheezing],
        yes_no_map[alcohol_consuming],
        yes_no_map[coughing],
        yes_no_map[shortness_of_breath],
        yes_no_map[swallowing_difficulty],
        yes_no_map[chest_pain]
    ]


    # Prediction button
    if st.button('Predict Lung Disease'):
        # Process user input to match model requirements
       
        lungcancer_diagnosis = ''
        lungcancer= lungcancer_model.predict([user_input])

        # Show the prediction result
        if lungcancer[0] == 1:
           lungcancer_diagnosis = 'The person do not have lung cancer'
        else:
           lungcancer_diagnosis = 'The person has lung cancer'

        st.success(lungcancer_diagnosis)
    
         
         
    #typhoid prediction
if selected == 'Typhoid Test':
    st.title('Predicting typhoid')

    # Collecting user inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0)
    with col2:
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    with col3:
        symptoms_severity = st.selectbox('Symptoms Severity', ['Mild', 'Moderate', 'Severe'])

    with col1:
        hemoglobin = st.number_input('Hemoglobin (g/dL)', min_value=0.0, step=0.1)
    with col2:
        platelet_count = st.number_input('Platelet Count', min_value=0)
    with col3:
        blood_culture_bacteria = st.selectbox('Blood Culture Bacteria', ['Present', 'Absent'])

    with col1:
        urine_culture_bacteria = st.selectbox('Urine Culture Bacteria', ['Present', 'Absent'])
    with col2:
        calcium = st.number_input('Calcium (mg/dL)', min_value=0.0, step=0.1)
    with col3:
        potassium = st.number_input('Potassium (mmol/L)', min_value=0.0, step=0.1)

    with col1:
        current_medication = st.text_input('Current Medication')
    with col2:
        treatment_duration = st.number_input('Treatment Duration (in months)', min_value=0)
    with col3:
        typhoid_history=st.selectbox('Typhoid History', ['yes', 'no'])

        
    # Mapping categorical inputs
    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    symptoms_severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2}
    bacteria_presence_map = {'Present': 1, 'Absent': 0}
    typhoid_history_map={'yes': 1, 'no': 0}

    # Processing user input
    user_input = [
        age,
        gender_map[gender],
        symptoms_severity_map[symptoms_severity],
        hemoglobin,
        platelet_count,
        bacteria_presence_map[blood_culture_bacteria],
        bacteria_presence_map[urine_culture_bacteria],
        typhoid_history_map[typhoid_history],
        calcium,
        potassium,
        len(current_medication),  # Using the length of medication text as a feature
        treatment_duration
    ]

    # Prediction button
    if st.button('Typhoid Test'):
        # Process user input to match model requirements
        
        # Placeholder for model prediction
        typhoid_test_diagnosis = ''
        

        kidney_disease = typhoid_model.predict([user_input])  # Replace with `processed_input` if preprocessing is needed
        if kidney_disease[0] == 1:
            typhoid_test_diagnosis = 'The person has typhoid'
        else:
            typhoid_test_diagnosis = 'The person does not have typhoid'
        
        # Display the result
        st.success(typhoid_test_diagnosis)

    #brain stroke 
if selected == 'Brain Stroke':
    st.title('Brain Stroke Prediction ')

    # Collecting user inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    with col2:
        age = st.number_input('Age', min_value=0, max_value=120, help="Enter the age of the individual")
    with col3:
        hypertension = st.selectbox('Hypertension', ['Yes', 'No'])

    with col1:
        heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
    with col2:
        ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
    with col3:
        work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Government', 'Children', 'Other'])

    with col1:
        Residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
    with col2:
        avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, format="%.1f", help="Enter the average glucose level")
    with col3:
        bmi = st.number_input('BMI', min_value=0.0, max_value=60.0, format="%.1f", help="Enter BMI (Body Mass Index)")

    with col1:
        smoking_status = st.selectbox('Smoking Status', ['Never smoked', 'Formerly smoked', 'Smokes', 'Unknown'])

    # Mapping inputs
    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    yes_no_map = {'Yes': 1, 'No': 0}
    work_type_map = {'Private': 0, 'Self-employed': 1, 'Government': 2, 'Children': 3, 'Other': 4}
    residence_map = {'Urban': 0, 'Rural': 1}
    smoking_status_map = {'Never smoked': 0, 'Formerly smoked': 1, 'Smokes': 2, 'Unknown': 3}

    user_input = [
        gender_map[gender],
        age,
        yes_no_map[hypertension],
        yes_no_map[heart_disease],
        yes_no_map[ever_married],
        work_type_map[work_type],
        residence_map[Residence_type],
        avg_glucose_level,
        bmi,
        smoking_status_map[smoking_status]
    ]

    # Prediction button
    if st.button('Predict Brain Stroke'):
       
            

            prediction = brainstroke_model.predict([user_input])[0]

            # Display result
            if prediction == 1:
                st.success('you are at risk of brain stroke.')
            else:
                st.success('you are not at risk of brain stroke')
    

      # thyroid check
      
if selected == 'Thyroid Test':  

    st.title('Thyroid Test')

    # Collecting user inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, help="Enter the age of the individual")

    with col2:
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])

    with col3:
        smoking = st.selectbox('Smoking', ['Yes', 'No'])

    with col1:
        hx_smoking = st.selectbox('Hx Smoking', ['Yes', 'No'])

    with col2:
        hx_radiotherapy = st.selectbox('Hx Radiotherapy', ['Yes', 'No'])

    with col3:
        thyroid_function = st.selectbox('Thyroid Function', ['Normal', 'Abnormal'])

    with col1:
        physical_examination = st.selectbox('Physical Examination', ['Normal', 'Abnormal'])

    with col2:
        adenopathy = st.selectbox('Adenopathy', ['Yes', 'No'])

    with col3:
        pathology = st.selectbox('Pathology', ['Positive', 'Negative'])

    with col1:
        focality = st.selectbox('Focality', ['Unifocal', 'Multifocal'])

    with col2:
        risk = st.selectbox('Risk', ['High', 'Medium', 'Low'])

    with col3:
        t_stage = st.selectbox('T Stage', ['T1', 'T2', 'T3', 'T4'])

    with col1:
        n_stage = st.selectbox('N Stage', ['N0', 'N1', 'N2', 'N3'])

    with col2:
        m_stage = st.selectbox('M Stage', ['M0', 'M1'])

    with col3:
        cancer_stage = st.selectbox('Cancer Stage', ['I', 'II', 'III', 'IV'])

    with col1:
        response = st.selectbox('Response', ['Good', 'Average', 'Poor'])

    # Mapping categorical inputs to numeric values
    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    yes_no_map = {'Yes': 1, 'No': 0}
    thyroid_function_map = {'Normal': 0, 'Abnormal': 1}
    physical_examination_map = {'Normal': 0, 'Abnormal': 1}
    adenopathy_map = {'Yes': 1, 'No': 0}
    pathology_map = {'Positive': 1, 'Negative': 0}
    focality_map = {'Unifocal': 0, 'Multifocal': 1}
    risk_map = {'High': 2, 'Medium': 1, 'Low': 0}
    cancer_stage_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
    response_map = {'Good': 2, 'Average': 1, 'Poor': 0}
    t_stage_map = {'T1': 1, 'T2': 2, 'T3': 3}
    m_stage_map = {'M0':0, 'M1':1}
    n_stage_map = {'N0':0, 'N1':1, 'N2':2, 'N3':3}

    # Collecting the user inputs and applying the mappings
    user_input = [
        age,
        gender_map[gender],
        yes_no_map[smoking],
        yes_no_map[hx_smoking],
        yes_no_map[hx_radiotherapy],
        thyroid_function_map[thyroid_function],
        physical_examination_map[physical_examination],
        adenopathy_map[adenopathy],
        pathology_map[pathology],
        focality_map[focality],
        risk_map[risk],
        response_map[response],
        cancer_stage_map[cancer_stage],
        t_stage_map[t_stage],
        n_stage_map[n_stage],
        m_stage_map[m_stage]
    ]

    # Prediction button
    if st.button('Predict Thyroid Outcome'):
        # Use the model to make a prediction
        prediction = thyroid_model.predict([user_input])[0]

        # Display the prediction result
        if prediction == 1:
            st.success('The person is at risk of thyroid')
        else:
            st.success('The person is not at risk of thyroid')

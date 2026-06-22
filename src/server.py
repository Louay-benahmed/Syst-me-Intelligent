import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

app = Flask(__name__)

# Global variables to cache models
MODEL_RF = None
MODEL_SVM = None
TARGET_NAMES = {0: 'Classe 0: Rapide', 1: 'Classe 1: Standard', 2: 'Classe 2: Critique'}


# =====================================================================
# MODEL TRAINING & INITIALIZATION (Adapted from your script)
# =====================================================================
def initialize_and_train_models():
    global MODEL_RF, MODEL_SVM
    print("[SERVER] Starting background model training sequence...")

    # Resolve file path
    dossier_script = os.path.dirname(os.path.abspath(__file__))  # Cible le dossier src/

    # On remonte d'un seul cran pour sortir de src/ et entrer dans 'Système Intelligent'
    dossier_racine = os.path.abspath(os.path.join(dossier_script, ".."))

    # On reconstruit proprement le chemin vers le fichier Excel
    filename = os.path.join(dossier_racine, "data", "Suivi délais d'exécution des échantillons 2025.xlsx")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Critical Error: Data file not found at {filename}")

    # 1. Load & Clean
    df = pd.read_excel(filename, header=3, engine='openpyxl')
    df = df.dropna(how='all', axis=1).dropna(subset=['N° échantillon'])

    mois_slice = df['N° échantillon'].astype(str).str.slice(4, 6)
    df['Mois_Extraction'] = pd.to_numeric(mois_slice, errors='coerce')
    df = df.dropna(subset=['Mois_Extraction'])
    df['Mois_Extraction'] = df['Mois_Extraction'].astype(int)

    # Calculate delays
    df['Date réception'] = pd.to_datetime(df['Date réception'], errors='coerce')
    df['Début analyse'] = pd.to_datetime(df['Début analyse'], errors='coerce')
    df['Fin Analyse'] = pd.to_datetime(df['Fin Analyse'], errors='coerce')
    df['Date édition du RA'] = pd.to_datetime(df['Date édition du RA'], errors='coerce')

    df['P1'] = (df['Début analyse'] - df['Date réception']).dt.days.clip(lower=0)
    df['DG'] = (df['Date édition du RA'] - df['Date réception']).dt.days.clip(lower=0)

    workload_map = df['Mois_Extraction'].value_counts().to_dict()
    df['Workload_Mensuel'] = df['Mois_Extraction'].map(workload_map)

    # 2. Target Formulation
    def assign_class(days):
        if pd.isna(days):
            return np.nan
        elif days <= 14:
            return 0
        elif days <= 30:
            return 1
        else:
            return 2

    df['Risque_Retard'] = df['DG'].apply(assign_class)
    df_ml = df.dropna(subset=['Risque_Retard', 'P1']).copy()
    df_ml['Risque_Retard'] = df_ml['Risque_Retard'].astype(int)

    # 3. Features Matrix & Training
    features = ['Mois_Extraction', 'P1', 'Workload_Mensuel']
    X = df_ml[features]
    y = df_ml['Risque_Retard']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Random Forest
    print("[SERVER] Fitting Random Forest model...")
    MODEL_RF = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced'))
    ])
    MODEL_RF.fit(X_train, y_train)

    # Train SVM with Optimization
    print("[SERVER] Optimizing and Fitting SVM model...")
    pipeline_svm_tuning = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=42))
    ])
    param_grid = {'classifier__C': [1, 10, 100], 'classifier__gamma': ['scale', 'auto', 0.1]}
    grid_search_svm = GridSearchCV(pipeline_svm_tuning, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search_svm.fit(X_train, y_train)

    best_params = grid_search_svm.best_params_
    optimized_svc = SVC(
        kernel='rbf', C=best_params['classifier__C'],
        gamma=best_params['classifier__gamma'],
        class_weight='balanced', random_state=42
    )

    MODEL_SVM = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', CalibratedClassifierCV(optimized_svc, ensemble=False, cv=5))
    ])
    MODEL_SVM.fit(X_train, y_train)
    print("[SERVER] All models are fully trained and active!")


# =====================================================================
# WEB ROUTES & ENDPOINTS
# =====================================================================

@app.route('/')
def home():
    # Serves the HTML frontend interface directly
    with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html'), 'r', encoding='utf-8') as f:
        html_content = f.read()
    return render_template_string(html_content)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract inputs from request JSON payload
        data = request.json
        mois = int(data.get('mois'))
        p1 = int(data.get('p1'))
        workload = int(data.get('workload'))
        model_choice = data.get('model', 'rf')  # 'rf' or 'svm'

        # Convert input parameters into a structured DataFrame matching feature schemas
        input_data = pd.DataFrame([[mois, p1, workload]], columns=['Mois_Extraction', 'P1', 'Workload_Mensuel'])

        # Elect selected model pipeline
        selected_pipeline = MODEL_SVM if model_choice == 'svm' else MODEL_RF

        # Compute inferences
        prediction = int(selected_pipeline.predict(input_data)[0])
        probabilities = selected_pipeline.predict_proba(input_data)[0]
        confidence = float(np.max(probabilities))

        return jsonify({
            'success': True,
            'prediction_class': prediction,
            'prediction_label': TARGET_NAMES[prediction],
            'confidence': confidence,
            'all_probabilities': {TARGET_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)}
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    # Initialize and fit backend ML models before firing up the web server
    initialize_and_train_models()
    # Run server locally on http://127.0.0.1:5000/
    app.run(debug=True, port=5000)
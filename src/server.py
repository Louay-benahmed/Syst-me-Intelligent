import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

TARGET_NAMES = {0: 'Classe 0: Rapide', 1: 'Classe 1: Standard', 2: 'Classe 2: Critique'}

# ⚡ CHARGEMENT INSTANTANÉ DES MODÈLES PRÉ-ENTRAÎNÉS (Bypass les timeouts Vercel)
dossier_script = os.path.dirname(os.path.abspath(__file__))
path_rf = os.path.join(dossier_script, 'model_rf.pkl')
path_svm = os.path.join(dossier_script, 'model_svm.pkl')

if os.path.exists(path_rf) and os.path.exists(path_svm):
    MODEL_RF = joblib.load(path_rf)
    MODEL_SVM = joblib.load(path_svm)
    print("[SERVER] Modèles d'IA sérialisés chargés avec succès.")
else:
    MODEL_RF = None
    MODEL_SVM = None
    print("[WARNING] Attention : Les fichiers .pkl sont introuvables. Lancez d'abord main.py sur votre PC.")


# =====================================================================
# ROUTES ET ENDPOINTS WEB
# =====================================================================
@app.route('/')
def home():
    # Sert l'interface HTML directement depuis le dossier templates/
    with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html'), 'r', encoding='utf-8') as f:
        html_content = f.read()
    return render_template_string(html_content)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if MODEL_RF is None or MODEL_SVM is None:
            return jsonify({'success': False, 'error': 'Les modèles ne sont pas encore prêts sur le serveur.'}), 500

        # Récupération des données du formulaire JSON frontend
        data = request.json
        mois = int(data.get('mois'))
        p1 = int(data.get('p1'))
        workload = int(data.get('workload'))
        model_choice = data.get('model', 'rf')

        # Structuration pour l'inférence conforme au pipeline d'entraînement
        input_data = pd.DataFrame([[mois, p1, workload]], columns=['Mois_Extraction', 'P1', 'Workload_Mensuel'])

        # Sélection de l'architecture choisie
        selected_pipeline = MODEL_SVM if model_choice == 'svm' else MODEL_RF

        # Inférence IA directe
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
    # Utilisé uniquement pour l'exécution et les tests en local (http://127.0.0.1:5000/)
    app.run(debug=True, port=5000)
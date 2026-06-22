import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

# Sélection des modèles conformément aux exigences académiques
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

# Configuration du style graphique pour le rapport d'analyse
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]


# =====================================================================
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES (Lecture Excel .xlsx)
# =====================================================================
def load_and_clean_data(filepath):
    print("[1/6] Chargement et nettoyage des données Excel en cours...")
    df = pd.read_excel(filepath, header=3, engine='openpyxl')

    df = df.dropna(how='all', axis=1)
    df = df.dropna(subset=['N° échantillon'])

    mois_slice = df['N° échantillon'].astype(str).str.slice(4, 6)
    df['Mois_Extraction'] = pd.to_numeric(mois_slice, errors='coerce')

    df = df.dropna(subset=['Mois_Extraction'])
    df['Mois_Extraction'] = df['Mois_Extraction'].astype(int)

    date_cols = ['Date réception', 'Début analyse', 'Fin Analyse', 'Date jugement', 'Date édition du RA',
                 'Date Facture']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['P1'] = (df['Début analyse'] - df['Date réception']).dt.days
    df['P2'] = (df['Date Facture'] - df['Date réception']).dt.days
    df['P3'] = (df['Fin Analyse'] - df['Début analyse']).dt.days
    df['P4'] = (df['Fin Analyse'] - df['Date réception']).dt.days
    df['P5'] = (df['Date jugement'] - df['Fin Analyse']).dt.days
    df['P6'] = (df['Date jugement'] - df['Date réception']).dt.days
    df['P7'] = (df['Date édition du RA'] - df['Date jugement']).dt.days
    df['P8'] = (df['Date édition du RA'] - df['Fin Analyse']).dt.days
    df['DG'] = (df['Date édition du RA'] - df['Date réception']).dt.days

    delay_features = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'DG']
    for feat in delay_features:
        df[feat] = df[feat].clip(lower=0)

    workload_map = df['Mois_Extraction'].value_counts().to_dict()
    df['Workload_Mensuel'] = df['Mois_Extraction'].map(workload_map)

    return df


# =====================================================================
# 2. FORMULATION DU PROBLÈME IA
# =====================================================================
def operational_target_definition(df):
    print("[2/6] Définition des classes de risque opérationnel...")

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

    print("\n--- Répartition des classes cibles obtenues (%) ---")
    print(df_ml['Risque_Retard'].value_counts(normalize=True) * 100)
    return df_ml


# =====================================================================
# 3. PROTOCOLE EXPÉRIMENTAL ET PIPELINES DE MACHINE LEARNING
# =====================================================================
def run_experimental_pipeline(df_ml):
    print("[3/6] Configuration du protocole de validation et optimisation...")

    features = ['Mois_Extraction', 'P1', 'Workload_Mensuel']
    X = df_ml[features]
    y = df_ml['Risque_Retard']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline_rf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced'))
    ])

    pipeline_svm_tuning = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=42))
    ])

    param_grid = {
        'classifier__C': [1, 10, 100],
        'classifier__gamma': ['scale', 'auto', 0.1]
    }

    print("Recherche des meilleurs hyperparamètres (C, Gamma) pour le SVM en cours...")
    grid_search_svm = GridSearchCV(pipeline_svm_tuning, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search_svm.fit(X_train, y_train)

    best_params = grid_search_svm.best_params_
    print(
        f"Meilleurs paramètres trouvés -> C: {best_params['classifier__C']}, Gamma: {best_params['classifier__gamma']}")

    optimized_svc = SVC(
        kernel='rbf',
        C=best_params['classifier__C'],
        gamma=best_params['classifier__gamma'],
        class_weight='balanced',
        random_state=42
    )

    pipeline_svm = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', CalibratedClassifierCV(optimized_svc, ensemble=False, cv=5))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- Validation Croisée Stratifiée (Macro F1-Score) ---")
    score_rf = cross_val_score(pipeline_rf, X_train, y_train, cv=cv, scoring='f1_macro')
    score_svm = cross_val_score(pipeline_svm, X_train, y_train, cv=cv, scoring='f1_macro')

    print(f"Random Forest - F1 Moyen : {score_rf.mean():.4f} (+/- {score_rf.std():.4f})")
    print(f"SVM Optimisé  - F1 Moyen : {score_svm.mean():.4f} (+/- {score_svm.std():.4f})")

    pipeline_rf.fit(X_train, y_train)
    pipeline_svm.fit(X_train, y_train)

    return pipeline_rf, pipeline_svm, X_test, y_test


# =====================================================================
# 4. ÉVALUATION DES PERFORMANCES CLASSIQUES
# =====================================================================
def evaluate_models(model_rf, model_svm, X_test, y_test):
    print("[4/6] Évaluation des performances sur l'ensemble de Test...")
    target_names = ['Classe 0: Rapide', 'Classe 1: Standard', 'Classe 2: Critique']

    for name, model in [("Random Forest", model_rf), ("SVM (RBF)", model_svm)]:
        y_pred = model.predict(X_test)
        print(f"\n================ MÉTRIQUES DE PERFORMANCE : {name} ================")
        print(classification_report(y_test, y_pred, target_names=target_names))

        kappa = cohen_kappa_score(y_test, y_pred)
        print(f"Indice Kappa de Cohen : {kappa:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title(f"Matrice de Confusion - {name}")
        plt.ylabel('Vérité Terrain (Réel)')
        plt.xlabel('Prédiction du Modèle')
        plt.tight_layout()
        plt.show()


# =====================================================================
# 5. ANALYSE PROFONDE DE LA CONFIANCE DES PRÉDICTIONS
# =====================================================================
def perform_confidence_analysis(model, X_test, y_test, model_name):
    print(f"[5/6] Extraction et diagnostic des scores de confiance pour {model_name}...")
    probas = model.predict_proba(X_test)
    predictions = model.predict(X_test)
    confiances = np.max(probas, axis=1)

    df_conf = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': predictions,
        'Confidence': confiances,
        'Is_Correct': (y_test == predictions)
    })

    print(f"\n--- Statistiques Descriptives de la Confiance ({model_name}) ---")
    print(df_conf['Confidence'].describe())

    mean_conf_correct = df_conf[df_conf['Is_Correct'] == True]['Confidence'].mean()
    mean_conf_wrong = df_conf[df_conf['Is_Correct'] == False]['Confidence'].mean()
    print(f"Confiance moyenne pour les prédictions CORRECTES : {mean_conf_correct:.4f}")
    print(f"Confiance moyenne pour les prédictions ERRONÉES  : {mean_conf_wrong:.4f}")

    plt.figure(figsize=(10, 5))
    sns.histplot(data=df_conf, x='Confidence', hue='Is_Correct', multiple='stack', bins=20, palette='viridis', kde=True)
    plt.title(f"Distribution des Taux de Confiance des Prédictions - {model_name}")
    plt.xlabel("Score de Confiance (Probabilité Maximale)")
    plt.ylabel("Nombre d'échantillons")
    plt.tight_layout()
    plt.show()

    doute_zone = df_conf[df_conf['Confidence'] < 0.55]
    print(
        f"Volume de prédictions en zone de doute critique (< 55% de confiance) : {len(doute_zone)} dossiers ({len(doute_zone) / len(df_conf) * 100:.2f}%)")

    return df_conf


# =====================================================================
# 6. EXÉCUTION DU SCRIPT & SÉRIALISATION DES MODÈLES DANS SRC/
# =====================================================================
if __name__ == "__main__":
    dossier_script = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.abspath(
        os.path.join(dossier_script, "..", "data", "Suivi délais d'exécution des échantillons 2025.xlsx"))

    print(f"Recherche locale du fichier Excel dans le projet : {filename}")

    if not os.path.exists(filename):
        print("\n[ERREUR CRITIQUE] Le fichier Excel spécifié est introuvable.")
    else:
        df_cleaned = load_and_clean_data(filename)
        df_ml = operational_target_definition(df_cleaned)
        model_rf, model_svm, X_test, y_test = run_experimental_pipeline(df_ml)

        evaluate_models(model_rf, model_svm, X_test, y_test)
        df_confidence_rf = perform_confidence_analysis(model_rf, X_test, y_test, "Random Forest")
        df_confidence_svm = perform_confidence_analysis(model_svm, X_test, y_test, "SVM")

        # 💾 EXPORTATION SÉCURISÉE DES PIPELINES EN FICHIERS CACHE
        print("\n[6/6] Sérialisation et sauvegarde des modèles d'architecture IA...")
        joblib.dump(model_rf, os.path.join(dossier_script, 'model_rf.pkl'))
        joblib.dump(model_svm, os.path.join(dossier_script, 'model_svm.pkl'))

        print("Fichiers 'model_rf.pkl' et 'model_svm.pkl' créés avec succès dans src/ !")
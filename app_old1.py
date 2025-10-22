from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional
import os

# === Initialisation de l'application ===
app = FastAPI(title="API Prédiction Durée Intervention", version="1.0")

# Configuration CORS pour permettre les requêtes du frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, remplacez par votre domaine Hostinger
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Chargement du modèle ===
print("Chargement du modèle...")
try:
    model_path = os.path.join(os.path.dirname(__file__), "modele_duree.pkl")
    model_data = joblib.load(model_path)
    classifier = model_data["classifier"]
    regressors = model_data["regressors"]
    scalers = model_data["scalers"]
    feature_cols = model_data["feature_cols"]
    print(f"✅ Modèle chargé avec succès! ({len(feature_cols)} features)")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    raise

# === Chargement des données de référence ===
try:
    csv_path = os.path.join(os.path.dirname(__file__), "AH_chir_final.csv")
    class_path = os.path.join(os.path.dirname(__file__), "bdd_traitee_1_classified.csv")
    
    df = pd.read_csv(csv_path, sep=';')
    df_class = pd.read_csv(class_path, sep=',')
    
    # Liste des codes DP uniques
    dp_codes = sorted(df['dp'].dropna().unique().tolist())
    
    # Liste des actes CCAM uniques
    actes_ccam = sorted(df['acte_classant'].dropna().unique().tolist())
    
    # === Dictionnaire des spécialités simplifiées ===
    categorie_dict = {
        'Chirurgie digestive': 'Digestif',
        'Chirurgie générale': 'Digestif',
        'Gastroentérologie': 'Digestif',
        'Chirurgie orthopédique et traumatologique': 'Orthopédique',
        'Orthopédie': 'Orthopédique',
        'Chirurgie vasculaire': 'Vasculaire',
        'Chirurgie cardiaque': 'Vasculaire',
        'Cardiologie': 'Vasculaire',
        'Chirurgie thoracique et cardiovasculaire': 'Vasculaire',
        'Urologie': 'Urologue',
        'Gynécologie-Obstétrique': 'Gynéco',
        'Gynécologie-obstétrique': 'Gynéco',
        'ORL et chirurgie cervico-faciale': 'ORL',
        'Oto-rhino-laryngologie (ORL)': 'ORL',
        'Ophtalmologie': 'Ophtalmo',
        'Chirurgie maxillo-faciale et stomatologie': 'Stomatologie',
        'Odontologie': 'Stomatologie',
        'Stomatologie': 'Stomatologie',
        'Chirurgie plastique': 'Plasticien',
        'Chirurgie de la main': 'Plasticien',
        'Chirurgie du rachis': 'Orthopédique',
        'Chirurgie thoracique': 'Vasculaire',
        'Chirurgie endocrinienne': 'Digestif',
        'Néphrologie': 'Digestif',
        'Anesthésiologie-réanimation chirurgicale': 'Digestif',
        'Radiologie interventionnelle': 'Digestif',
        'Radiologie': 'Digestif',
        'Neurochirurgie': 'Orthopédique',
        'Pneumologie': 'Digestif',
        'Dermatologie': 'Dermato',
        'Psychiatrie': 'Digestif'
    }
    
    # Créer mapping acte -> spécialité depuis df_class
    df_class['Classification_list'] = df_class['Classification'].fillna('').apply(
        lambda x: [c.strip() for c in x.split(';')] if x else []
    )
    df_class_exploded = df_class.explode('Classification_list')
    df_class_exploded['medecin_type'] = df_class_exploded['Classification_list'].map(categorie_dict)
    
    acte_to_specialite = dict(zip(
        df_class_exploded['acte_classant'],
        df_class_exploded['medecin_type']
    ))
    
    # Charger le LabelEncoder pour dp
    from sklearn.preprocessing import LabelEncoder
    le_dp = LabelEncoder()
    le_dp.fit(df['dp'].fillna('INCONNU'))
    
    print(f"✅ {len(dp_codes)} codes DP chargés")
    print(f"✅ {len(actes_ccam)} actes CCAM chargés")
    
except Exception as e:
    print(f"⚠️ Erreur lors du chargement des données de référence: {e}")
    dp_codes = []
    actes_ccam = []
    acte_to_specialite = {}
    le_dp = None

# === Modèles de données ===
class PredictionRequest(BaseModel):
    age: int
    sexe: int  # 0 = Femme, 1 = Homme
    acte_ccam: str
    dp: str

class PredictionResponse(BaseModel):
    categorie_predite: str
    duree_predite: float
    confiance: str
    details: dict

# === Endpoints ===
@app.get("/")
def read_root():
    return {
        "message": "API Prédiction Durée Intervention",
        "status": "running",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Prédire la durée",
            "/codes_dp": "GET - Liste des codes DP",
            "/actes_ccam": "GET - Liste des actes CCAM",
            "/health": "GET - Vérifier l'état de l'API"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "features_count": len(feature_cols),
        "dp_codes_count": len(dp_codes),
        "actes_ccam_count": len(actes_ccam)
    }

@app.get("/codes_dp")
def get_dp_codes():
    """Retourne la liste de tous les codes DP disponibles"""
    return {"codes": dp_codes[:100]}  # Limité à 100 pour l'affichage

@app.get("/actes_ccam")
def get_actes_ccam():
    """Retourne la liste de tous les actes CCAM disponibles"""
    return {"actes": actes_ccam[:100]}  # Limité à 100 pour l'affichage

@app.post("/predict", response_model=PredictionResponse)
def predict_duration(request: PredictionRequest):
    """
    Prédire la durée d'intervention
    """
    try:
        # Vérifier les données d'entrée
        if request.age < 0 or request.age > 120:
            raise HTTPException(status_code=400, detail="Âge invalide")
        
        if request.sexe not in [0, 1]:
            raise HTTPException(status_code=400, detail="Sexe invalide (0=F, 1=H)")
        
        # Déterminer la spécialité à partir de l'acte CCAM
        specialite = acte_to_specialite.get(request.acte_ccam, 'Digestif')  # Défaut
        
        # Encoder le code DP
        try:
            if le_dp is not None:
                dp_encoded = le_dp.transform([request.dp])[0]
            else:
                dp_encoded = 0
        except:
            dp_encoded = 0  # Valeur par défaut si code inconnu
        
        # Créer le vecteur de features
        patient_data = {
            'age': request.age,
            'sexe': request.sexe,
            'medecin_type_Urologue': 1 if specialite == 'Urologue' else 0,
            'medecin_type_ORL': 1 if specialite == 'ORL' else 0,
            'medecin_type_Ophtalmo': 1 if specialite == 'Ophtalmo' else 0,
            'medecin_type_Vasculaire': 1 if specialite == 'Vasculaire' else 0,
            'medecin_type_Digestif': 1 if specialite == 'Digestif' else 0,
            'medecin_type_Gynéco': 1 if specialite == 'Gynéco' else 0,
            'medecin_type_Orthopédique': 1 if specialite == 'Orthopédique' else 0,
            'medecin_type_Stomatologie': 1 if specialite == 'Stomatologie' else 0,
            'medecin_type_Dermato': 1 if specialite == 'Dermato' else 0,
            'medecin_type_Plasticien': 1 if specialite == 'Plasticien' else 0,
            'dp_encoded': dp_encoded
        }
        
        # Créer DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Ajouter les colonnes manquantes avec 0
        for col in feature_cols:
            if col not in patient_df.columns:
                patient_df[col] = 0
        
        # Réorganiser dans le bon ordre
        patient_df = patient_df[feature_cols]
        
        # 1. Prédire la catégorie
        categorie_predite = classifier.predict(patient_df)[0]
        proba = classifier.predict_proba(patient_df)[0]
        confiance_score = max(proba)
        
        # Déterminer le niveau de confiance
        if confiance_score > 0.8:
            confiance = "Haute"
        elif confiance_score > 0.6:
            confiance = "Moyenne"
        else:
            confiance = "Faible"
        
        # 2. Prédire la durée avec le régresseur approprié
        if regressors[categorie_predite] is not None:
            patient_scaled = scalers[categorie_predite].transform(patient_df)
            duree_predite = regressors[categorie_predite].predict(patient_scaled)[0]
            duree_predite = max(1.0, round(duree_predite, 1))  # Minimum 1 jour
        else:
            # Fallback
            duree_predite = 3.0 if categorie_predite == "Court" else (5.0 if categorie_predite == "Moyen" else 10.0)
        
        # Préparer la réponse
        response = PredictionResponse(
            categorie_predite=categorie_predite,
            duree_predite=duree_predite,
            confiance=confiance,
            details={
                "specialite_detectee": specialite,
                "age": request.age,
                "sexe": "Homme" if request.sexe == 1 else "Femme",
                "acte_ccam": request.acte_ccam,
                "code_dp": request.dp,
                "probabilites": {
                    "Court": float(proba[0] if len(proba) > 0 else 0),
                    "Long": float(proba[1] if len(proba) > 1 else 0),
                    "Moyen": float(proba[2] if len(proba) > 2 else 0)
                }
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

# === Lancement de l'application ===
if __name__ == "__main__":
    import uvicorn
    # Pour le développement local
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

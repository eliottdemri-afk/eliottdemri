"""
API FastAPI pour prédiction de durée d'intervention + Optimisation de planning hospitalier
Intègre la génération de patients, les algorithmes V4 (Recuit Simulé) et V5 (Génétique)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import os
import json
import io
import copy
import random
import math
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import asyncio
from collections import defaultdict

# ============================================================================
# INITIALISATION DE L'APPLICATION
# ============================================================================

app = FastAPI(title="API Prédiction & Optimisation Hospitalière", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, remplacer par votre domaine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CHARGEMENT DU MODÈLE DE PRÉDICTION
# ============================================================================

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
    classifier, regressors, scalers, feature_cols = None, None, None, None

# Chargement de la base de données pour le mapping CCAM -> Spécialité
try:
    bdd_path = os.path.join(os.path.dirname(__file__), "bdd_traitee_1_classified.csv")
    df_bdd = pd.read_csv(bdd_path)
    CCAM_TO_SPE = dict(zip(df_bdd["acte_ccam"], df_bdd["specialite"]))
    print(f"✅ Base de données chargée: {len(CCAM_TO_SPE)} codes CCAM")
except Exception as e:
    print(f"❌ Erreur chargement BDD: {e}")
    CCAM_TO_SPE = {}

# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================

@dataclass
class Patient:
    """Patient avec durée d'opération"""
    id: int
    age: int
    sexe: int
    ccam: str
    dp: str
    duree_sejour_predite: float
    duree_op: float
    specialite: str
    date_souhaitee: int

@dataclass
class Operation:
    """Représentation interne pour l'algorithme"""
    id_op: int
    duree_op: float
    duree_rum: float
    spe_rss: List[str]
    date: int
    medecin: Optional[int] = None
    salle: Optional[int] = None
    jour: Optional[int] = None

class Solution:
    """Représente une solution de planning"""
    def __init__(self, operations: List[Operation], config: dict):
        self.operations = operations
        self.config = config
        self.cout = float('inf')

    def copy(self):
        return Solution([copy.deepcopy(op) for op in self.operations], self.config.copy())

# ============================================================================
# MODÈLES PYDANTIC POUR L'API
# ============================================================================

class PredictionInput(BaseModel):
    age: int
    sexe: int
    acte_ccam: str
    dp: str

class PredictionOutput(BaseModel):
    duree_sejour_predite: float
    classe_predite: int

class HospitalConfig(BaseModel):
    """Configuration de l'hôpital"""
    nb_lits: int = 100
    nb_salles: int = 6
    t_max_salle: float = 9.0  # heures
    t_max_medecin: float = 8.0  # heures
    capacite_weekend: float = 0.5  # pourcentage
    date_debut: str = "2026-01-01"
    date_fin: str = "2026-12-31"
    medecins: List[Dict] = [
        {"specialite": "Chirurgie digestive", "nombre": 7, "duree_moyenne": 150},
        {"specialite": "Gynécologie-obstétrique", "nombre": 3, "duree_moyenne": 80},
        {"specialite": "Neurochirurgie", "nombre": 1, "duree_moyenne": 135},
        {"specialite": "ORL et chirurgie cervico-faciale", "nombre": 2, "duree_moyenne": 130},
        {"specialite": "Ophtalmologie", "nombre": 1, "duree_moyenne": 110},
        {"specialite": "Chirurgie orthopédique et traumatologique", "nombre": 8, "duree_moyenne": 150},
        {"specialite": "Chirurgie plastique", "nombre": 1, "duree_moyenne": 158},
        {"specialite": "Stomatologie", "nombre": 3, "duree_moyenne": 160},
        {"specialite": "Urologie", "nombre": 5, "duree_moyenne": 90},
        {"specialite": "Chirurgie vasculaire", "nombre": 5, "duree_moyenne": 145},
    ]

class AlgoParams(BaseModel):
    """Paramètres des algorithmes"""
    algo_type: str = "genetic"  # "genetic" ou "annealing"
    # Paramètres génétique
    taille_population: int = 20
    nb_generations: int = 50
    taux_mutation: float = 0.2
    taux_elitisme: float = 0.1
    # Paramètres recuit simulé
    t_init: float = 100.0
    t_min: float = 0.1
    alpha: float = 0.9
    max_iter: int = 25

class OptimizationRequest(BaseModel):
    """Requête d'optimisation complète"""
    nb_patients: int
    hospital_config: HospitalConfig
    algo_params: AlgoParams

class AddPatientRequest(BaseModel):
    """Ajout d'un patient après optimisation"""
    age: int
    sexe: int
    acte_ccam: str
    dp: str
    date_souhaitee: str
    planning_actuel: List[Dict]
    hospital_config: HospitalConfig

# ============================================================================
# LISTE DES ACTES CCAM AUTORISÉS
# ============================================================================

LISTE_ACTES_AUTORISES = [
    "HMFC004","FCFA009","JDDB007","JGFE023","LFAA002","JCAE001","DELF005","FCCA001",
    "NHDA009","DEKA001","EBFA002","FCCA001","JJFC010","DELF005","HMFC004","JKFA027",
    "QZMA001","QZFA038","JKDC001","NEKA020","JCAE001","HHFA002","HFFC018","QEFA020",
    "JDFE001","JGFA005","QEGA004","JJFC003","JCAE001","JDFE001","JANE005","HMFC004",
    "NEKA009","JFFC001","JKDC001","JJFC006","HHCC007","JKFC002","NEKA020","JKDC001",
    "NEKA005","HHFA016","KDQA001","KCFA005","LFAA002","JGFE023","JCGE005","JDFE001",
    "LFFA002","MJFA013","LMMC002","GDFA008","EDFA007","JEMA023","HPPC002","FCPA001",
    "DEMA001","NEKA009","NFFA001","HBFA013","HEPA002","JDFE001","NEKA009","NFFA001",
    "GDFA007","QEGA004","FCPA001","NFFA001","DEKA001","JCFE007","MJFA013","HFCC004",
    "HHFC024","CDFA002","LFFA002","JFFC001","HHFC024","MJMC002","MJMC002","MJFA013",
]

# ============================================================================
# GÉNÉRATION DE PATIENTS
# ============================================================================

def generer_patients(nb_patients: int, date_debut: str, date_fin: str) -> pd.DataFrame:
    """
    Génère un DataFrame de patients avec age, sexe, acte_ccam, dp
    """
    patients = []

    # Convertir les dates en timestamp
    date_debut_dt = datetime.strptime(date_debut, "%Y-%m-%d")
    date_fin_dt = datetime.strptime(date_fin, "%Y-%m-%d")
    nb_jours = (date_fin_dt - date_debut_dt).days

    for i in range(nb_patients):
        age = random.randint(18, 90)
        sexe = random.choice([1, 2])
        acte_ccam = random.choice(LISTE_ACTES_AUTORISES)

        # DP basé sur CCAM
        dp_options = {
            "HMFC004": ["I251", "I350", "I119"],
            "JDDB007": ["M171", "M170", "M179"],
            "LFAA002": ["C509", "C500", "C504"],
        }
        dp = random.choice(dp_options.get(acte_ccam, ["Z511", "Z485", "R101"]))

        # Date souhaitée aléatoire
        date_souhaitee = random.randint(0, nb_jours)

        patients.append({
            "age": age,
            "sexe": sexe,
            "acte_ccam": acte_ccam,
            "dp": dp,
            "date_souhaitee": date_souhaitee
        })

    return pd.DataFrame(patients)

# ============================================================================
# PRÉDICTION DE DURÉE DE SÉJOUR
# ============================================================================

def predire_duree_sejour(df_patients: pd.DataFrame) -> pd.DataFrame:
    """
    Prédit la durée de séjour pour chaque patient
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    df = df_patients.copy()

    # Créer les features manquantes
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].fillna(0)

    # Prédiction de la classe
    df["classe_predite"] = classifier.predict(X)

    # Prédiction de la durée pour chaque classe
    durees = []
    for idx, row in df.iterrows():
        classe = int(row["classe_predite"])
        if classe in regressors:
            X_scaled = scalers[classe].transform([X.iloc[idx]])
            duree = regressors[classe].predict(X_scaled)[0]
            durees.append(max(1, duree))
        else:
            durees.append(5.0)

    df["duree_sejour_predite"] = durees
    return df

# ============================================================================
# ALGORITHME GÉNÉTIQUE V5
# ============================================================================

class AlgorithmeGenetique:
    def __init__(self, patients: List[Patient], config: HospitalConfig, params: AlgoParams):
        self.patients = patients
        self.config = config
        self.params = params

        # Configuration hôpital
        self.nb_salles = config.nb_salles
        self.nb_lits = config.nb_lits
        self.t_max_salle = config.t_max_salle * 60  # en minutes
        self.t_max_medecin = config.t_max_medecin * 60
        self.capacite_weekend = config.capacite_weekend

        # Calculer nb_jours_max
        date_debut = datetime.strptime(config.date_debut, "%Y-%m-%d")
        date_fin = datetime.strptime(config.date_fin, "%Y-%m-%d")
        self.nb_jours_max = (date_fin - date_debut).days

        # Médecins par spécialité
        self.medecins_par_spe = {}
        for m in config.medecins:
            self.medecins_par_spe[m["specialite"]] = m["nombre"]

        # Vacances scolaires 2025-2026 (dates françaises)
        self.vacances = self._generer_vacances()

    def _generer_vacances(self) -> List[Tuple[int, int]]:
        """Génère les périodes de vacances scolaires"""
        date_ref = datetime.strptime(self.config.date_debut, "%Y-%m-%d")
        vacances_dates = [
            ("2026-02-07", "2026-02-23"),  # Hiver
            ("2026-04-04", "2026-04-20"),  # Printemps
            ("2026-07-04", "2026-08-31"),  # Été
            ("2026-10-24", "2026-11-09"),  # Toussaint
            ("2026-12-19", "2027-01-04"),  # Noël
        ]

        periodes = []
        for debut_str, fin_str in vacances_dates:
            try:
                debut = (datetime.strptime(debut_str, "%Y-%m-%d") - date_ref).days
                fin = (datetime.strptime(fin_str, "%Y-%m-%d") - date_ref).days
                if 0 <= debut < self.nb_jours_max:
                    periodes.append((max(0, debut), min(fin, self.nb_jours_max)))
            except:
                pass

        return periodes

    def _est_weekend(self, jour: int) -> bool:
        date_ref = datetime.strptime(self.config.date_debut, "%Y-%m-%d")
        date = date_ref + timedelta(days=jour)
        return date.weekday() >= 5

    def _est_vacances(self, jour: int) -> bool:
        return any(debut <= jour <= fin for debut, fin in self.vacances)

    def _generer_solution_initiale(self) -> Solution:
        """Génère une solution initiale aléatoire"""
        operations = []
        for patient in self.patients:
            op = Operation(
                id_op=patient.id,
                duree_op=patient.duree_op,
                duree_rum=patient.duree_sejour_predite,
                spe_rss=[patient.specialite],
                date=patient.date_souhaitee,
                medecin=None,
                salle=None,
                jour=None
            )
            operations.append(op)

        solution = Solution(operations, asdict(self.config))
        self._affecter_ressources(solution)
        return solution

    def _affecter_ressources(self, solution: Solution):
        """Affecte médecins et salles aux opérations"""
        # Réinitialiser
        for op in solution.operations:
            op.medecin = None
            op.salle = None
            op.jour = None

        # Trier par date souhaitée
        operations_triees = sorted(solution.operations, key=lambda x: x.date)

        # Structures de suivi
        occupation_medecins = defaultdict(lambda: defaultdict(float))
        occupation_salles = defaultdict(lambda: defaultdict(float))
        lits_occupes = defaultdict(int)

        for op in operations_triees:
            planifie = False
            specialite = op.spe_rss[0] if op.spe_rss else "Autre"
            nb_medecins = self.medecins_par_spe.get(specialite, 1)

            # Essayer de planifier sur 30 jours max
            for delta in range(31):
                jour_test = op.date + delta
                if jour_test >= self.nb_jours_max:
                    break

                # Capacité réduite weekend
                capacite = self.capacite_weekend if self._est_weekend(jour_test) else 1.0
                t_max_salle_jour = self.t_max_salle * capacite
                t_max_medecin_jour = self.t_max_medecin * capacite

                # Vérifier lits disponibles
                lits_requis = int(op.duree_rum) + 1
                if lits_occupes[jour_test] + lits_requis > self.nb_lits:
                    continue

                # Chercher médecin disponible
                medecin_trouve = None
                for m in range(nb_medecins):
                    if occupation_medecins[jour_test][m] + op.duree_op <= t_max_medecin_jour:
                        medecin_trouve = m
                        break

                if medecin_trouve is None:
                    continue

                # Chercher salle disponible
                salle_trouvee = None

                # Contrainte: Neuro/Ortho UNIQUEMENT salle 6 (dernière)
                if specialite in ["Neurochirurgie", "Chirurgie orthopédique et traumatologique"]:
                    salle_test = self.nb_salles - 1
                    if occupation_salles[jour_test][salle_test] + op.duree_op <= t_max_salle_jour:
                        salle_trouvee = salle_test
                else:
                    # Autres spécialités: salles 0 à nb_salles-2
                    for s in range(self.nb_salles - 1):
                        if occupation_salles[jour_test][s] + op.duree_op <= t_max_salle_jour:
                            salle_trouvee = s
                            break

                if salle_trouvee is not None:
                    # Planifier
                    op.jour = jour_test
                    op.medecin = medecin_trouve
                    op.salle = salle_trouvee

                    occupation_medecins[jour_test][medecin_trouve] += op.duree_op
                    occupation_salles[jour_test][salle_trouvee] += op.duree_op

                    for d in range(lits_requis):
                        if jour_test + d < self.nb_jours_max:
                            lits_occupes[jour_test + d] += 1

                    planifie = True
                    break

            if not planifie:
                # Forcer planification à la fin
                op.jour = self.nb_jours_max - 1
                op.medecin = 0
                op.salle = 0

    def _calculer_cout(self, solution: Solution) -> float:
        """Calcule le coût de la solution"""
        penalite_date = 0
        penalite_non_planifie = 0

        for op in solution.operations:
            if op.jour is None:
                penalite_non_planifie += 10000
            else:
                ecart = abs(op.jour - op.date)
                penalite_date += ecart * 10

        # Calculer variation des lits (RMSD)
        lits_par_jour = defaultdict(int)
        for op in solution.operations:
            if op.jour is not None:
                duree_rum = int(op.duree_rum) + 1
                for d in range(duree_rum):
                    if op.jour + d < self.nb_jours_max:
                        lits_par_jour[op.jour + d] += 1

        occupations = list(lits_par_jour.values())
        if occupations:
            moyenne = sum(occupations) / len(occupations)
            rmsd = np.sqrt(sum((x - moyenne)**2 for x in occupations) / len(occupations))
        else:
            rmsd = 0

        cout_total = penalite_date + penalite_non_planifie + rmsd * 100
        solution.cout = cout_total
        return cout_total

    def _selection(self, population: List[Solution]) -> Solution:
        """Sélection par tournoi"""
        tournoi = random.sample(population, min(3, len(population)))
        return min(tournoi, key=lambda s: s.cout)

    def _croisement(self, parent1: Solution, parent2: Solution) -> Solution:
        """Croisement à un point"""
        enfant = parent1.copy()
        point = len(enfant.operations) // 2

        for i in range(point, len(enfant.operations)):
            if i < len(parent2.operations):
                enfant.operations[i].date = parent2.operations[i].date

        self._affecter_ressources(enfant)
        return enfant

    def _mutation(self, solution: Solution):
        """Mutation: modifier dates aléatoirement"""
        for op in solution.operations:
            if random.random() < self.params.taux_mutation:
                op.date = random.randint(0, self.nb_jours_max - 1)

        self._affecter_ressources(solution)

    def optimiser(self, progress_callback=None) -> Solution:
        """Lance l'optimisation génétique"""
        # Initialiser population
        population = []
        for _ in range(self.params.taille_population):
            sol = self._generer_solution_initiale()
            self._calculer_cout(sol)
            population.append(sol)

        meilleure_solution = min(population, key=lambda s: s.cout)

        # Évolution
        for gen in range(self.params.nb_generations):
            # Élitisme
            population_triee = sorted(population, key=lambda s: s.cout)
            nb_elites = int(self.params.taille_population * self.params.taux_elitisme)
            nouvelle_population = population_triee[:nb_elites]

            # Génération descendants
            while len(nouvelle_population) < self.params.taille_population:
                parent1 = self._selection(population)
                parent2 = self._selection(population)
                enfant = self._croisement(parent1, parent2)
                self._mutation(enfant)
                self._calculer_cout(enfant)
                nouvelle_population.append(enfant)

            population = nouvelle_population
            meilleure = min(population, key=lambda s: s.cout)

            if meilleure.cout < meilleure_solution.cout:
                meilleure_solution = meilleure

            if progress_callback:
                progress = int((gen + 1) / self.params.nb_generations * 100)
                progress_callback(progress, f"Génération {gen+1}/{self.params.nb_generations}")

        return meilleure_solution

# ============================================================================
# ALGORITHME RECUIT SIMULÉ V4
# ============================================================================

class RecuitSimule:
    def __init__(self, patients: List[Patient], config: HospitalConfig, params: AlgoParams):
        self.patients = patients
        self.config = config
        self.params = params

        # Configuration similaire à AlgorithmeGenetique
        self.nb_salles = config.nb_salles
        self.nb_lits = config.nb_lits
        self.t_max_salle = config.t_max_salle * 60
        self.t_max_medecin = config.t_max_medecin * 60
        self.capacite_weekend = config.capacite_weekend

        date_debut = datetime.strptime(config.date_debut, "%Y-%m-%d")
        date_fin = datetime.strptime(config.date_fin, "%Y-%m-%d")
        self.nb_jours_max = (date_fin - date_debut).days

        self.medecins_par_spe = {}
        for m in config.medecins:
            self.medecins_par_spe[m["specialite"]] = m["nombre"]

        self.vacances = self._generer_vacances()

    def _generer_vacances(self) -> List[Tuple[int, int]]:
        date_ref = datetime.strptime(self.config.date_debut, "%Y-%m-%d")
        vacances_dates = [
            ("2026-02-07", "2026-02-23"),
            ("2026-04-04", "2026-04-20"),
            ("2026-07-04", "2026-08-31"),
            ("2026-10-24", "2026-11-09"),
            ("2026-12-19", "2027-01-04"),
        ]

        periodes = []
        for debut_str, fin_str in vacances_dates:
            try:
                debut = (datetime.strptime(debut_str, "%Y-%m-%d") - date_ref).days
                fin = (datetime.strptime(fin_str, "%Y-%m-%d") - date_ref).days
                if 0 <= debut < self.nb_jours_max:
                    periodes.append((max(0, debut), min(fin, self.nb_jours_max)))
            except:
                pass

        return periodes

    def _est_weekend(self, jour: int) -> bool:
        date_ref = datetime.strptime(self.config.date_debut, "%Y-%m-%d")
        date = date_ref + timedelta(days=jour)
        return date.weekday() >= 5

    def _generer_solution_initiale(self) -> Solution:
        operations = []
        for patient in self.patients:
            op = Operation(
                id_op=patient.id,
                duree_op=patient.duree_op,
                duree_rum=patient.duree_sejour_predite,
                spe_rss=[patient.specialite],
                date=patient.date_souhaitee,
                medecin=None,
                salle=None,
                jour=None
            )
            operations.append(op)

        solution = Solution(operations, asdict(self.config))
        self._affecter_ressources(solution)
        return solution

    def _affecter_ressources(self, solution: Solution):
        for op in solution.operations:
            op.medecin = None
            op.salle = None
            op.jour = None

        operations_triees = sorted(solution.operations, key=lambda x: x.date)

        occupation_medecins = defaultdict(lambda: defaultdict(float))
        occupation_salles = defaultdict(lambda: defaultdict(float))
        lits_occupes = defaultdict(int)

        for op in operations_triees:
            planifie = False
            specialite = op.spe_rss[0] if op.spe_rss else "Autre"
            nb_medecins = self.medecins_par_spe.get(specialite, 1)

            for delta in range(31):
                jour_test = op.date + delta
                if jour_test >= self.nb_jours_max:
                    break

                capacite = self.capacite_weekend if self._est_weekend(jour_test) else 1.0
                t_max_salle_jour = self.t_max_salle * capacite
                t_max_medecin_jour = self.t_max_medecin * capacite

                lits_requis = int(op.duree_rum) + 1
                if lits_occupes[jour_test] + lits_requis > self.nb_lits:
                    continue

                medecin_trouve = None
                for m in range(nb_medecins):
                    if occupation_medecins[jour_test][m] + op.duree_op <= t_max_medecin_jour:
                        medecin_trouve = m
                        break

                if medecin_trouve is None:
                    continue

                salle_trouvee = None

                if specialite in ["Neurochirurgie", "Chirurgie orthopédique et traumatologique"]:
                    salle_test = self.nb_salles - 1
                    if occupation_salles[jour_test][salle_test] + op.duree_op <= t_max_salle_jour:
                        salle_trouvee = salle_test
                else:
                    for s in range(self.nb_salles - 1):
                        if occupation_salles[jour_test][s] + op.duree_op <= t_max_salle_jour:
                            salle_trouvee = s
                            break

                if salle_trouvee is not None:
                    op.jour = jour_test
                    op.medecin = medecin_trouve
                    op.salle = salle_trouvee

                    occupation_medecins[jour_test][medecin_trouve] += op.duree_op
                    occupation_salles[jour_test][salle_trouvee] += op.duree_op

                    for d in range(lits_requis):
                        if jour_test + d < self.nb_jours_max:
                            lits_occupes[jour_test + d] += 1

                    planifie = True
                    break

            if not planifie:
                op.jour = self.nb_jours_max - 1
                op.medecin = 0
                op.salle = 0

    def _calculer_cout(self, solution: Solution) -> float:
        penalite_date = 0
        penalite_non_planifie = 0

        for op in solution.operations:
            if op.jour is None:
                penalite_non_planifie += 10000
            else:
                ecart = abs(op.jour - op.date)
                penalite_date += ecart * 10

        lits_par_jour = defaultdict(int)
        for op in solution.operations:
            if op.jour is not None:
                duree_rum = int(op.duree_rum) + 1
                for d in range(duree_rum):
                    if op.jour + d < self.nb_jours_max:
                        lits_par_jour[op.jour + d] += 1

        occupations = list(lits_par_jour.values())
        if occupations:
            moyenne = sum(occupations) / len(occupations)
            rmsd = np.sqrt(sum((x - moyenne)**2 for x in occupations) / len(occupations))
        else:
            rmsd = 0

        cout_total = penalite_date + penalite_non_planifie + rmsd * 100
        solution.cout = cout_total
        return cout_total

    def _voisin(self, solution: Solution) -> Solution:
        nouvelle_solution = solution.copy()

        # Choisir opération aléatoire
        if nouvelle_solution.operations:
            op = random.choice(nouvelle_solution.operations)
            # Modifier légèrement la date
            op.date = max(0, min(self.nb_jours_max - 1, 
                                 op.date + random.randint(-7, 7)))

        self._affecter_ressources(nouvelle_solution)
        return nouvelle_solution

    def optimiser(self, progress_callback=None) -> Solution:
        # Solution initiale
        solution_courante = self._generer_solution_initiale()
        self._calculer_cout(solution_courante)
        meilleure_solution = solution_courante.copy()

        T = self.params.t_init
        iteration = 0

        while T > self.params.t_min:
            for _ in range(self.params.max_iter):
                nouvelle_solution = self._voisin(solution_courante)
                self._calculer_cout(nouvelle_solution)

                delta = nouvelle_solution.cout - solution_courante.cout

                if delta < 0 or random.random() < math.exp(-delta / T):
                    solution_courante = nouvelle_solution

                    if solution_courante.cout < meilleure_solution.cout:
                        meilleure_solution = solution_courante.copy()

                iteration += 1

            T *= self.params.alpha

            if progress_callback:
                progress = int((1 - T / self.params.t_init) * 100)
                progress_callback(progress, f"Température: {T:.2f}")

        return meilleure_solution

# ============================================================================
# ENDPOINTS API
# ============================================================================

@app.get("/")
def root():
    return {
        "message": "API Prédiction & Optimisation Hospitalière",
        "version": "2.0",
        "endpoints": ["/predict", "/optimize", "/add_patient", "/health"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "ccam_mapping": len(CCAM_TO_SPE) > 0
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_duree(data: PredictionInput):
    """Prédit la durée de séjour pour un patient"""
    try:
        df = pd.DataFrame([{
            "age": data.age,
            "sexe": data.sexe,
            "acte_ccam": data.acte_ccam,
            "dp": data.dp
        }])

        df_pred = predire_duree_sejour(df)

        return PredictionOutput(
            duree_sejour_predite=float(df_pred["duree_sejour_predite"].iloc[0]),
            classe_predite=int(df_pred["classe_predite"].iloc[0])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize")
async def optimize_planning(request: OptimizationRequest):
    """
    Lance l'optimisation du planning hospitalier
    """
    try:
        # 1. Générer patients
        df_patients = generer_patients(
            request.nb_patients,
            request.hospital_config.date_debut,
            request.hospital_config.date_fin
        )

        # 2. Prédire durées de séjour
        df_patients = predire_duree_sejour(df_patients)

        # 3. Créer objets Patient avec durées d'opération
        patients = []
        for idx, row in df_patients.iterrows():
            specialite = CCAM_TO_SPE.get(row["acte_ccam"], "Autre")

            # Trouver durée d'opération moyenne
            duree_op = 120  # par défaut
            for m in request.hospital_config.medecins:
                if m["specialite"] == specialite:
                    duree_op = m["duree_moyenne"]
                    break

            patient = Patient(
                id=idx,
                age=int(row["age"]),
                sexe=int(row["sexe"]),
                ccam=row["acte_ccam"],
                dp=row["dp"],
                duree_sejour_predite=float(row["duree_sejour_predite"]),
                duree_op=float(duree_op),
                specialite=specialite,
                date_souhaitee=int(row["date_souhaitee"])
            )
            patients.append(patient)

        # 4. Lancer l'algorithme choisi
        if request.algo_params.algo_type == "genetic":
            algo = AlgorithmeGenetique(patients, request.hospital_config, request.algo_params)
        else:
            algo = RecuitSimule(patients, request.hospital_config, request.algo_params)

        solution = algo.optimiser()

        # 5. Calculer métriques
        lits_par_jour = defaultdict(int)
        operations_par_jour = defaultdict(int)
        operations_par_specialite = defaultdict(int)

        for op in solution.operations:
            if op.jour is not None:
                operations_par_jour[op.jour] += 1

                patient = next(p for p in patients if p.id == op.id_op)
                operations_par_specialite[patient.specialite] += 1

                duree_rum = int(op.duree_rum) + 1
                for d in range(duree_rum):
                    lits_par_jour[op.jour + d] += 1

        # 6. Formater résultats
        planning = []
        date_debut = datetime.strptime(request.hospital_config.date_debut, "%Y-%m-%d")

        for op in solution.operations:
            if op.jour is not None:
                patient = next(p for p in patients if p.id == op.id_op)
                date_operation = date_debut + timedelta(days=op.jour)

                planning.append({
                    "patient_id": op.id_op,
                    "age": patient.age,
                    "sexe": patient.sexe,
                    "ccam": patient.ccam,
                    "dp": patient.dp,
                    "specialite": patient.specialite,
                    "date_operation": date_operation.strftime("%Y-%m-%d"),
                    "jour": op.jour,
                    "salle": op.salle,
                    "medecin": op.medecin,
                    "duree_op": op.duree_op,
                    "duree_sejour": op.duree_rum,
                    "date_souhaitee": patient.date_souhaitee,
                    "ecart_jours": op.jour - patient.date_souhaitee
                })

        # Métriques lits
        occupation_lits = []
        for jour in sorted(lits_par_jour.keys()):
            date = date_debut + timedelta(days=jour)
            occupation_lits.append({
                "jour": jour,
                "date": date.strftime("%Y-%m-%d"),
                "nb_lits": lits_par_jour[jour]
            })

        # Métriques opérations
        operations_jour = []
        for jour in sorted(operations_par_jour.keys()):
            date = date_debut + timedelta(days=jour)
            operations_jour.append({
                "jour": jour,
                "date": date.strftime("%Y-%m-%d"),
                "nb_operations": operations_par_jour[jour]
            })

        # Métriques spécialités
        specialites = [
            {"specialite": spe, "nb_operations": nb}
            for spe, nb in operations_par_specialite.items()
        ]

        # Calcul RMSD
        occupations = list(lits_par_jour.values())
        if occupations:
            moyenne = sum(occupations) / len(occupations)
            rmsd = float(np.sqrt(sum((x - moyenne)**2 for x in occupations) / len(occupations)))
        else:
            rmsd = 0.0

        return {
            "success": True,
            "algorithm": request.algo_params.algo_type,
            "nb_patients": len(patients),
            "nb_planifies": len(planning),
            "cout_total": float(solution.cout),
            "rmsd_lits": rmsd,
            "planning": planning,
            "occupation_lits": occupation_lits,
            "operations_par_jour": operations_jour,
            "operations_par_specialite": specialites
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur optimisation: {str(e)}")

@app.post("/add_patient")
async def add_patient_to_planning(request: AddPatientRequest):
    """
    Ajoute un patient à un planning existant
    """
    try:
        # 1. Prédire durée de séjour
        df_patient = pd.DataFrame([{
            "age": request.age,
            "sexe": request.sexe,
            "acte_ccam": request.acte_ccam,
            "dp": request.dp
        }])

        df_patient = predire_duree_sejour(df_patient)

        # 2. Créer objet Patient
        specialite = CCAM_TO_SPE.get(request.acte_ccam, "Autre")
        duree_op = 120
        for m in request.hospital_config.medecins:
            if m["specialite"] == specialite:
                duree_op = m["duree_moyenne"]
                break

        date_debut = datetime.strptime(request.hospital_config.date_debut, "%Y-%m-%d")
        date_souhaitee_dt = datetime.strptime(request.date_souhaitee, "%Y-%m-%d")
        jour_souhaite = (date_souhaitee_dt - date_debut).days

        nouveau_patient = Patient(
            id=999999,  # ID temporaire
            age=request.age,
            sexe=request.sexe,
            ccam=request.acte_ccam,
            dp=request.dp,
            duree_sejour_predite=float(df_patient["duree_sejour_predite"].iloc[0]),
            duree_op=float(duree_op),
            specialite=specialite,
            date_souhaitee=jour_souhaite
        )

        # 3. Reconstruire planning actuel
        patients_existants = []
        for p_dict in request.planning_actuel:
            p = Patient(
                id=p_dict["patient_id"],
                age=p_dict["age"],
                sexe=p_dict["sexe"],
                ccam=p_dict["ccam"],
                dp=p_dict["dp"],
                duree_sejour_predite=p_dict["duree_sejour"],
                duree_op=p_dict["duree_op"],
                specialite=p_dict["specialite"],
                date_souhaitee=p_dict["date_souhaitee"]
            )
            patients_existants.append(p)

        # 4. Ajouter nouveau patient
        patients_existants.append(nouveau_patient)

        # 5. Chercher un créneau (algorithme simple)
        config = request.hospital_config
        nb_salles = config.nb_salles
        t_max_salle = config.t_max_salle * 60
        t_max_medecin = config.t_max_medecin * 60

        # Reconstruire occupation
        occupation_salles = defaultdict(lambda: defaultdict(float))
        occupation_medecins = defaultdict(lambda: defaultdict(float))
        lits_occupes = defaultdict(int)

        for p_dict in request.planning_actuel:
            jour = p_dict["jour"]
            salle = p_dict["salle"]
            medecin = p_dict["medecin"]
            duree_op = p_dict["duree_op"]
            duree_rum = int(p_dict["duree_sejour"]) + 1

            occupation_salles[jour][salle] += duree_op
            occupation_medecins[jour][medecin] += duree_op

            for d in range(duree_rum):
                lits_occupes[jour + d] += 1

        # Chercher créneau pour nouveau patient
        date_fin = datetime.strptime(config.date_fin, "%Y-%m-%d")
        nb_jours_max = (date_fin - date_debut).days

        nb_medecins_spe = 1
        for m in config.medecins:
            if m["specialite"] == specialite:
                nb_medecins_spe = m["nombre"]
                break

        creneau_trouve = None

        for delta in range(31):
            jour_test = jour_souhaite + delta
            if jour_test >= nb_jours_max:
                break

            # Vérifier lits
            lits_requis = int(nouveau_patient.duree_sejour_predite) + 1
            if lits_occupes[jour_test] + lits_requis > config.nb_lits:
                continue

            # Chercher médecin
            medecin_dispo = None
            for m in range(nb_medecins_spe):
                if occupation_medecins[jour_test][m] + nouveau_patient.duree_op <= t_max_medecin:
                    medecin_dispo = m
                    break

            if medecin_dispo is None:
                continue

            # Chercher salle
            salle_dispo = None

            if specialite in ["Neurochirurgie", "Chirurgie orthopédique et traumatologique"]:
                salle_test = nb_salles - 1
                if occupation_salles[jour_test][salle_test] + nouveau_patient.duree_op <= t_max_salle:
                    salle_dispo = salle_test
            else:
                for s in range(nb_salles - 1):
                    if occupation_salles[jour_test][s] + nouveau_patient.duree_op <= t_max_salle:
                        salle_dispo = s
                        break

            if salle_dispo is not None:
                creneau_trouve = {
                    "jour": jour_test,
                    "date": (date_debut + timedelta(days=jour_test)).strftime("%Y-%m-%d"),
                    "salle": salle_dispo,
                    "medecin": medecin_dispo
                }
                break

        if creneau_trouve is None:
            return {
                "success": False,
                "message": "Aucun créneau disponible dans les 30 jours suivant la date souhaitée",
                "suggestion": "Veuillez ré-optimiser le planning ou augmenter les ressources"
            }

        # Ajouter au planning
        date_operation_dt = date_debut + timedelta(days=creneau_trouve["jour"])
        nouveau_planning_entry = {
            "patient_id": 999999,
            "age": nouveau_patient.age,
            "sexe": nouveau_patient.sexe,
            "ccam": nouveau_patient.ccam,
            "dp": nouveau_patient.dp,
            "specialite": nouveau_patient.specialite,
            "date_operation": date_operation_dt.strftime("%Y-%m-%d"),
            "jour": creneau_trouve["jour"],
            "salle": creneau_trouve["salle"],
            "medecin": creneau_trouve["medecin"],
            "duree_op": nouveau_patient.duree_op,
            "duree_sejour": nouveau_patient.duree_sejour_predite,
            "date_souhaitee": jour_souhaite,
            "ecart_jours": creneau_trouve["jour"] - jour_souhaite
        }

        return {
            "success": True,
            "message": "Patient ajouté avec succès",
            "nouveau_patient": nouveau_planning_entry,
            "creneau": creneau_trouve
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur ajout patient: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

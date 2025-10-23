# 🏥 Optimisation Planning Hospitalier


Système d'optimisation intelligente de planification hospitalière utilisant des algorithmes génétiques et de recuit simulé, couplé à un modèle de machine learning pour prédire les durées de séjour.



## 🎯 Vue d'ensemble

Ce projet résout le problème complexe de planification des opérations chirurgicales en tenant compte de multiples contraintes :

- **Ressources limitées** : salles d'opération, médecins, lits
- **Contraintes temporelles** : horaires de travail, week-ends, vacances scolaires
- **Objectifs contradictoires** : maximiser le nombre d'opérations tout en minimisant l'occupation des lits
- **Prédictions ML** : estimation automatique des durées de séjour basée sur les données CCAM

### 🎓 Contexte académique

Projet développé dans le cadre d'un travail académique sur l'optimisation combinatoire appliquée au domaine hospitalier.

## ✨ Fonctionnalités

### 🧬 Algorithmes d'optimisation
- **Algorithme génétique** : Évolution par sélection, croisement et mutation
- **Recuit simulé** : Exploration intelligente de l'espace des solutions
- **Mode asynchrone** : Gestion des optimisations longues via background tasks

### 🤖 Machine Learning
- **Modèle hybride** : Classification + Régression pour prédire les durées de séjour
- **Base de données CCAM** : 2910+ codes d'actes médicaux classifiés par spécialité
- **Features enrichies** : Âge, sexe, diagnostic principal, acte CCAM

### 📊 Interface utilisateur
- **Tableau de bord interactif** : Visualisation en temps réel des résultats
- **Graphiques Chart.js** : Occupation des lits, opérations par jour, répartition par spécialité
- **Export de données** : CSV et Excel
- **Progression asynchrone** : Barre de progression avec polling toutes les 2 secondes

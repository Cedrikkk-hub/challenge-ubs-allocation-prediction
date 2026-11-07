# Prédire le signe d'un rendement financier presque imprévisible — et pourquoi un bon score ne veut pas toujours dire un bon modèle

Étude de cas sur le **Challenge UBS 2026-2027** (plateforme [Challenge Data, ENS](https://challengedata.ens.fr/)), un challenge académique sponsorisé par CFM, Prairie, FSMP, CNRS, Fondation CDF, LVMH et Fondation Engie.

## Le problème

Pour chaque allocation d'actifs (une stratégie de trading), prédire le **signe** de son rendement futur : faut-il suivre l'allocation telle quelle, ou parier contre elle ? La métrique est l'Accuracy sur ce signe — seule la direction compte, pas l'ampleur.

Chaque ligne du jeu de données associe une date et une allocation, avec les 20 derniers rendements et volumes connus de cette allocation. Le signal est intrinsèquement très faible : la classe majoritaire représente déjà environ 52% des cas, et le score du notebook benchmark officiel (LightGBM) plafonne à **0.5079** — à peine au-dessus du hasard.

## Ce que j'ai fait

**1. Construit deux schémas de validation, volontairement différents, pour comparer leur fiabilité.**
Le benchmark officiel valide par K-Fold aléatoire sur les dates. Mais le jeu de test est entièrement postérieur au jeu d'entraînement (aucune date en commun) : un K-Fold mélangé fait donc de l'*interpolation*, alors que la vraie tâche est de l'*extrapolation* dans le futur. J'ai comparé ce schéma à un holdout chronologique strict (les N derniers jours d'entraînement comme validation). Résultat : les scores "optimistes" du K-Fold mélangé (jusqu'à +1.5 point au-dessus de la baseline) ne se vérifient plus du tout sous le schéma chronologique, où certains modèles ne battent même pas la classe majoritaire.

**2. Construit des features à historique causal strict**, par allocation et par groupe d'allocations : win-rate cumulé depuis le début de l'historique, moyennes/volatilités glissantes sur des fenêtres plus longues que les 20 jours fournis — toujours avec un `shift(1)` avant tout cumul, pour qu'aucune ligne n'utilise sa propre cible ni celle d'une ligne future. Pour le jeu de test (entièrement postérieur), la dernière valeur connue de chaque statistique est "gelée" et réappliquée telle quelle.

**3. Repéré une anomalie de classement, et vérifié qu'elle ne provenait pas de mon propre code.**
Le classement public affichait des scores jusqu'à 0.82, et le classement académique de ma cohorte avait un score de tête à 0.70 — très supérieurs à ce que suggérait toute analyse rigoureuse du signal disponible. Deux hypothèses de fuite de données ont été testées et réfutées empiriquement (jointure sur la ligne du jour suivant, biais directionnel par allocation). La confirmation est venue du **classement privé** (cette plateforme sépare un sous-ensemble de test privé, révélé seulement deux fois par an) : son meilleur score n'était que de 0.5504. L'écart de 27 points avec le classement public signe une exploitation du sous-ensemble public (probablement du *leaderboard probing*), pas un vrai modèle généralisable.

**4. Comparé plusieurs familles de modèles et d'ensembles**, en gardant toujours le schéma de validation chronologique comme juge de paix : régression vs classification directe du signe, ensemble des deux, RandomForest, ExtraTrees, stacking à deux niveaux avec méta-modèle entraîné sur un échantillon séparé.

## Résultats

| Étape | Accuracy (validation chronologique) | Score public réel |
|---|---|---|
| Benchmark officiel | — | 0.5079 |
| Modèles de base (régression / classification / ensemble) | ~52.2–52.5% | 0.5104 |
| + historique causal long par allocation | ~52.4% | **0.5131** |
| + historique par groupe, features de volume, réglage des hyperparamètres | ~52.6% | 0.5125 |

La dernière ligne illustre la leçon la plus intéressante de cet exercice : l'écart entre les deux dernières soumissions (0.5131 vs 0.5125) est plus petit que l'erreur-type attendue d'une mesure d'Accuracy sur un échantillon de cette taille. Les différences de quelques dixièmes de point entre modèles candidats, mesurées sur un seul découpage temporel, sont statistiquement indiscernables du bruit — sélectionner automatiquement "le modèle qui gagne sur ce holdout" est un biais de sélection classique (*winner's curse*), pas une garantie de généralisation. Un ensemble qui moyenne plusieurs modèles est plus robuste à ce bruit qu'un choix du meilleur candidat sur un seul split.

## Pourquoi un score "modeste" est en réalité un bon résultat ici

Rapporté au classement académique : progression du rang 10 au rang 8, avec un score nettement au-dessus du benchmark officiel et proche du meilleur score du classement privé (0.5504) — qui est la seule mesure non biaisée disponible sur ce problème. Les scores publics à 0.70–0.82 observés chez d'autres participants ne sont, selon toute vraisemblance, pas reproductibles sur données non vues.

## Stack technique

Python, pandas, numpy, scikit-learn, LightGBM.

## Reproduire

Les données appartiennent au challenge et ne sont pas incluses dans ce dépôt — elles sont téléchargeables (avec inscription) sur [challengedata.ens.fr](https://challengedata.ens.fr/participants/challenges/167/).

```bash
pip install pandas numpy scikit-learn lightgbm
python pipeline_allocation_challenge.py
```

Placez `X_train.csv`, `y_train.csv`, `X_test.csv`, `sample_submission.csv` dans le même dossier que le script (ou modifiez les chemins dans la section `CONFIG` en tête de fichier).

## Structure du dépôt

```
.
├── pipeline_allocation_challenge.py   # pipeline complet : features, validation, modèles, soumission
├── README.md
└── requirements.txt
```

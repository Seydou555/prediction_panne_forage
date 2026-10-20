# Exploration des données

Historique : 3000 lignes ; nouveau mois : 600 lignes.

## Qualité (historique)

|                          | type    |   manquants |   uniques |
|:-------------------------|:--------|------------:|----------:|
| age_ans                  | int64   |           0 |        29 |
| profondeur_m             | int64   |           0 |       120 |
| debit_m3h                | float64 |           0 |       155 |
| usagers_jour             | int64   |           0 |       807 |
| jours_depuis_maintenance | int64   |           0 |       703 |
| pannes_12m               | int64   |           0 |         6 |
| type_pompe               | str     |           0 |         3 |
| region                   | str     |           0 |         4 |
| panne_90j                | int64   |           0 |         2 |


## Statistiques descriptives (historique)

|                          |   count |   mean |    std |   min |   25% |   50% |    75% |   max |
|:-------------------------|--------:|-------:|-------:|------:|------:|------:|-------:|------:|
| age_ans                  |    3000 |  15.17 |   8.42 |   1   |   8   |  15   |  23    |  29   |
| profondeur_m             |    3000 |  88.68 |  34.95 |  30   |  58   |  88   | 119    | 149   |
| debit_m3h                |    3000 |   4.44 |   2.81 |   0.1 |   2.4 |   3.9 |   5.93 |  24.4 |
| usagers_jour             |    3000 | 486.47 | 237.47 |  80   | 281   | 483.5 | 692    | 899   |
| jours_depuis_maintenance |    3000 | 354.23 | 207.19 |   0   | 176   | 350.5 | 530    | 719   |
| pannes_12m               |    3000 |   0.77 |   0.87 |   0   |   0   |   1   |   1    |   5   |


## Cible
Taux de panne historique : 0.363 ; nouveau mois : 0.447

## Taux de panne par modalité

| type_pompe   |   taux_panne |   effectif |
|:-------------|-------------:|-----------:|
| manuelle     |        0.43  |       1471 |
| solaire      |        0.255 |        920 |
| thermique    |        0.361 |        609 |


| region   |   taux_panne |   effectif |
|:---------|-------------:|-----------:|
| Centre   |        0.378 |        768 |
| Est      |        0.363 |        743 |
| Nord     |        0.379 |        759 |
| Sud      |        0.329 |        730 |


|   pannes_12m |   taux_panne |   effectif |
|-------------:|-------------:|-----------:|
|            0 |        0.276 |       1382 |
|            1 |        0.386 |       1079 |
|            2 |        0.496 |        411 |
|            3 |        0.633 |        109 |
|            4 |        0.944 |         18 |
|            5 |        1     |          1 |


## Corrélation avec la cible

|                          |   corr |
|:-------------------------|-------:|
| age_ans                  |  0.237 |
| profondeur_m             | -0.003 |
| debit_m3h                | -0.069 |
| usagers_jour             |  0.128 |
| jours_depuis_maintenance |  0.205 |
| pannes_12m               |  0.215 |


## Historique vs nouveau mois

|    | variable                 |   moy_hist |   moy_new |   max_hist |   max_new |   ecart_moy_% |
|---:|:-------------------------|-----------:|----------:|-----------:|----------:|--------------:|
|  0 | age_ans                  |      15.17 |     14.81 |       29   |      29   |         -2.37 |
|  1 | profondeur_m             |      88.68 |     91    |      149   |     149   |          2.61 |
|  2 | debit_m3h                |       4.44 |      4.26 |       24.4 |      19.3 |         -3.98 |
|  3 | usagers_jour             |     486.47 |    642.55 |      899   |    1212   |         32.08 |
|  4 | jours_depuis_maintenance |     354.23 |    353.02 |      719   |     719   |         -0.34 |
|  5 | pannes_12m               |       0.77 |      1.14 |        5   |       6   |         47.72 |
|  6 | panne_90j                |       0.36 |      0.45 |        1   |       1   |         23.16 |


## Audit de fuite

- Doublons dans l'historique : 0

- Lignes communes historique / nouveau mois : 0


AUC de chaque variable seule (arbre de profondeur 3, CV 5 plis) :


|                          |   auc |
|:-------------------------|------:|
| age_ans                  | 0.633 |
| pannes_12m               | 0.611 |
| jours_depuis_maintenance | 0.601 |
| type_pompe_manuelle      | 0.572 |
| type_pompe_solaire       | 0.571 |
| usagers_jour             | 0.562 |
| debit_m3h                | 0.536 |
| region_Sud               | 0.518 |
| region_Nord              | 0.509 |
| region_Centre            | 0.508 |
| type_pompe_thermique     | 0.492 |
| region_Est               | 0.486 |
| profondeur_m             | 0.467 |


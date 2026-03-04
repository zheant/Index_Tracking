# CLAUDE.md — Historique de travail sur Index_Tracking

## Vue d'ensemble du projet

Projet de **réplication d'indice boursier par sous-portefeuille** (index tracking).
L'objectif est de sélectionner K actions parmi les constituants d'un indice (SP500, Russell 3000)
et de leur affecter des poids optimaux pour minimiser l'erreur de tracking hors-échantillon.

---

## Architecture du projet

```
Index_Tracking/
├── main.py                          # Point d'entrée CLI (optimisation + sauvegarde)
├── prafa/
│   ├── universe.py                  # Gestion de l'univers de stocks et nettoyage des données
│   ├── portfolio.py                 # Classe Portfolio + Solution (orchestre les solveurs)
│   ├── quob.py                      # Solveur QUOB (via ReplicaTOR binaire)
│   ├── gurobi.py                    # Solveur Gurobi (MILP)
│   └── matrix_utils.py             # Calcul des matrices de distance (dcor, Pearson)
├── scripts/
│   ├── analyze_results.py           # Backtest et visualisation des résultats
│   ├── prepare_russell_constituents.py  # Normalisation des CSV constituants Russell
│   └── download_wrds_russell_data.py    # Téléchargement des données WRDS
├── financial_data/
│   ├── sp500/                       # Données SP500 (returns_stocks.csv, returns_index.csv, constituants/)
│   └── russell3000/                 # Données Russell 3000 (idem)
└── results/                         # Portefeuilles sauvegardés (.pkl) + analyses (.png)
```

---

## Flux d'exécution principal

1. **`main.py`** : définit les fenêtres d'entraînement glissantes, instancie `Portfolio(Universe(args))`, appelle `rebalance_portfolio()` à chaque date de rééquilibrage, puis sauvegarde le résultat en `.pkl`.
2. **`universe.py`** : charge les données de rendement, charge les constituants annuels, applique le pipeline de nettoyage (`data_cleaning()`).
3. **`portfolio.py`** : classe `Solution` qui dispatch vers le bon solveur (`quob`, `gurobi`, `lagrange_*`).
4. **`scripts/analyze_results.py`** : recharge les `.pkl`, reconstruit les séries de rendements hors-échantillon, calcule tracking error et MAE, génère 5 graphiques.

---

## Paramètres CLI importants (`main.py`)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--index` | `sp500` | Indice cible (`sp500`, `russell3000`) |
| `--solution_name` | `quob` | Solveur (`quob`, `gurobi`, `lagrange_*`) |
| `--cardinality` | `50` | Nombre de stocks K dans le sous-portefeuille |
| `--T` | `3` | Durée de la fenêtre d'entraînement (années) |
| `--rebalancing` | `12` | Fréquence de rééquilibrage (mois) |
| `--missing_policy` | `auto` | `legacy` (SP500, fillna=0) ou `strict` (Russell, nettoyage complet) |
| `--reconstitution_month` | `7` | Mois d'activation de la nouvelle composition (Russell : juillet) |
| `--max_missing_frac` | `0.10` | Fraction max de NaN tolérée par stock (training) |
| `--min_trading_frac` | `0.50` | Fraction min de jours à rendement non nul (filtre liquidité) |
| `--winsor_sigma` | `3.0` | Seuil de winsorisation en σ (training uniquement) |
| `--hard_clip` | `1.0` | Clip absolu des rendements aberrants en ±% (training + test) |
| `--distance_method` | `dcor` | Métrique de distance (`dcor` ou `pearson`) |
| `--time_limit` | `300` | Limite de temps pour ReplicaTOR et Gurobi (secondes) |

---

## Évolution du code — Historique de travail

### Phase 1 : Mise en place initiale (novembre 2025)
- Initialisation du repo avec les données SP500.
- Structure de base : `Universe`, `Portfolio`, `Solution`.
- Solveurs `lagrange_full`, `lagrange_ours`, `lagrange_forward`, `lagrange_backward`.
- Intégration de Gurobi comme solveur MILP de référence.
- Configuration du `.gitignore` pour exclure la licence Gurobi locale.

### Phase 2 : Intégration du solveur QUOB / ReplicaTOR (décembre 2025)
- Ajout du solveur `QUOB` dans `prafa/quob.py` utilisant le binaire `ReplicaTOR` (parallel tempering).
- `QUOB` : sélection de K médoïdes via optimisation combinatoire, puis pondération QP (`SLSQP`).
- Écriture des fichiers `dist_matrix.d`, `dist_matrix.adj`, `dist_matrix.params` dans un répertoire temporaire isolé par instance (sécurité pour les runs parallèles).
- `matrix_dcor()` : matrice de corrélation de distance (distance correlation de Székely).
- `matrix_simplecor()` : alternative basée sur la corrélation de Pearson.
- Paramètre `--distance_method` (`dcor`/`pearson`) exposé dans le CLI.

### Phase 3 : Extension au Russell 3000 (janvier 2026 — semaine du 7)
- **Extraction de `matrix_utils.py`** : `compute_dcor_matrix()` et `compute_simplecor_matrix()` déplacées dans un module partagé utilisé par QUOB et Gurobi.
- **Politique de données manquantes** :
  - `legacy` (SP500) : `fillna(0)`, aucun filtrage, comportement historique préservé.
  - `strict` (Russell 3000) : pipeline complet de nettoyage multi-étapes.
- **Pipeline `data_cleaning()` (strict)** :
  1. Suppression des stocks trop lacunaires (`max_missing_frac`, training uniquement).
  2. Remplissage des NaN par 0 (jours sans transaction = rendement nul).
  3. Hard-clip des rendements aberrants (`±hard_clip`) — training ET test, sans look-ahead.
  4. Filtre de liquidité (`min_trading_frac`, training uniquement).
  5. Winsorisation par σ par stock, calculée sur les jours de trading uniquement (training uniquement).
  6. Suppression des lignes où l'indice est manquant.
  7. Vérification de cardinalité minimale.
- **Reconstitution annuelle du Russell** :
  - Les fichiers `{year}.csv` représentent la composition post-reconstitution de juin {year}, active à partir de juillet.
  - `_effective_constituent_year()` : gère le décalage de mois.
  - `update_stock_list()` : charge le bon fichier constituant selon la date effective.
  - ~~Intersection des univers constituants sur toute la fenêtre~~ → supprimée en Phase 7 (biais de survivant).
- **`_select_constituent_year()`** : fallback sur l'année disponible la plus récente si l'année exacte n'existe pas.
- **Reset `self.year = -1`** en fin de `new_universe()` pour forcer le rechargement au prochain appel.
- **`_get_constituents_for_date()`** : méthode sans effet de bord pour consulter les constituants.

### Phase 4 : Améliorations du backtest et analyse (janvier 2026 — semaines 9-10)
- **`analyze_results.py`** reécrit en script CLI autonome :
  - Charge les `.pkl` multi-méthodes simultanément.
  - `extract_timeseries()` : roll-forward hors-échantillon entre rebalancements consécutifs.
  - `_align_weights()` : aligne les poids sauvegardés vers les colonnes de l'univers de test.
  - Fenêtre de test : `portfolio_key + 1 BDay` → `dates[i+1] - 1 BDay` (évite le chevauchement training/test).
  - Gestion des NaN en test : `fillna(0)` (position cash pour les actions délistées).
  - Suppression du filtre `min_presence` en test (introduisait du look-ahead).
- **Vérification de calendrier** : hash SHA-256 du calendrier enregistré lors du training, comparé au calendrier effectif du test pour détecter les incohérences.
- **5 graphiques générés** :
  - `cumulative_returns.png`
  - `tracking_errors.png`
  - `tracking_absolute_errors.png`
  - `cumulative_and_absolute_errors.png`
  - `error_distributions.png`
- **`_build_args()`** : reconstruction du namespace d'arguments pour correspondre exactement aux paramètres utilisés lors de la génération des portefeuilles.
- Paramètre `--missing_policy auto` : choisit `legacy` pour SP500 et `strict` sinon.

### Phase 5 : Robustesse QUOB (janvier 2026)
- Répertoire temporaire isolé par instance QUOB (`tempfile.mkdtemp`) → safe pour runs parallèles.
- Méthode `_cleanup()` appelée dans le bloc `finally` de `get_weights()`.
- Parser de secours `_parse_medoids_from_output()` si ReplicaTOR ne crée pas le fichier `.soln.txt`.
- Vérification explicite de l'exécutabilité du binaire ReplicaTOR.
- Renormalisation défensive des poids SLSQP en cas de dérive numérique.
- Gurobi : retour d'un portefeuille nul avec warning si aucun poids n'est retourné (timeout, infaisabilité).

---

## Décisions architecturales clés

### Anti-look-ahead en entraînement
- L'univers est basé sur la composition à `end_datetime` (date de décision d'investissement).
  Les stocks entrés en cours de training sont inclus; les sortants sont absents du fichier constituant.
  Les stocks avec historique insuffisant sont éliminés par `max_missing_frac`.
- Winsorisation et filtre de liquidité calculés uniquement sur les données d'entraînement.

### Anti-look-ahead en test
- `training=False` dans `new_universe()` : supprime les filtres dépendant des statistiques futures.
- Hard-clip est la seule transformation appliquée en test (valeur fixe, pas de statistique de fenêtre).
- Fenêtre de test démarre 1 jour ouvré après la dernière date d'entraînement.

### Séparation des politiques de données manquantes
- `legacy` : comportement original SP500 conservé pour reproductibilité.
- `strict` : nettoyage complet pour Russell 3000 où les small/micro caps ont beaucoup de NaN.

### Fichiers de constituants
- SP500 : format inconnu (géré automatiquement par `_discover_constituent_files()`).
- Russell 3000 : fichiers `{year}.csv` (permno uniquement), découverts sous `constituants/` ou `constituants_raw/`.

---

## Scripts utilitaires

### `scripts/prepare_russell_constituents.py`
Normalise les CSV bruts Russell 3000 (colonne `permno` obligatoire) vers `financial_data/russell3000/constituants/`.
Génère aussi `all_permnos.csv` avec l'union de tous les permnos observés.

### `scripts/download_wrds_russell_data.py`
Télécharge les données de rendement depuis WRDS (Wharton Research Data Services).

### Phase 7 : Correction de deux erreurs méthodologiques majeures (4 mars 2026)

**Fix #1 — `prafa/universe.py` : référence `end_datetime` pour l'univers de training**

- **Problème** : l'univers d'entraînement était basé sur la composition à `start_datetime` + intersection
  de toutes les reconstitutions intermédiaires. Cela excluait les stocks entrés en cours de training,
  créant un biais de survivant — l'univers de réplication s'écartait structurellement de la composition
  réelle de l'indice au moment de la décision d'investissement.
- **Correctif** : `ref_date = end_datetime if training else start_datetime` dans `new_universe()`.
  Suppression du bloc d'intersection inter-années (devenu inutile : les sortants sont naturellement
  absents du fichier constituant à `end_datetime`; les lacunaires sont éliminés par `max_missing_frac`).
- **Mode test** inchangé : `ref_date = start_datetime` (composition active sans look-ahead).

**Fix #2 — `scripts/analyze_results.py` : suppression du calendar hash check incorrect**

- **Problème** : le `calendar_hash` stocké dans le `.pkl` est le hash du calendrier d'**entraînement**.
  Il était comparé au hash du calendrier de **test** — deux périodes distinctes par construction.
  Ce check déclenchait un faux warning systématique et ne détectait jamais une vraie incohérence.
- **Correctif** : suppression du bloc `if saved_calendar_hash:` dans `extract_timeseries()`.

**Validation** : run SP500, K=50, QUOB, 2 fenêtres. Logs confirmés :
- Univers training `[2016-01-02 → 2019-01-02]` : 505 stocks chargés avec `ref=2019-01-02` ✓
- Aucun warning `Calendar mismatch` dans `analyze_results.py` ✓

---

### Phase 6 : Contrainte de rendement moyen + analyse du time_limit (4 mars 2026)

**Constat** : après le premier run Russell 3000 (QUOB, K=300, pearson, strict, 10 800s), le portefeuille
présente une **sous-performance cumulée** par rapport à l'indice malgré une tracking error faible.

**Cause principale identifiée** : l'objectif QP `Σ(r_p - r_i)²` minimise la variance de l'erreur mais
n'impose pas `E[r_p] = E[r_i]`. Un biais négatif systématique peut persister.

**Correctif implémenté** — contrainte de rendement moyen dans `calc_weights()` (QUOB et Gurobi) :
- Ajout de la contrainte `μ_stocks · w = μ_index` dans le SLSQP.
- Fallback sur `sum=1` seul si SLSQP ne converge pas avec la contrainte (avec warning).
- Fichiers modifiés : `prafa/quob.py` et `prafa/gurobi.py`.

**Analyse du time_limit** :
- Un ancien log (`Index_Tracking_Russell3000`) révèle que pour n=2605, K=300 et time_limit=60s,
  l'algorithme a exécuté **489 rounds** et a atteint la limite de temps (pas convergé naturellement).
- Extrapolation : à 10 800s → ~88 000 rounds, toujours très loin du round_limit (100M).
- L'algorithme n'a probablement **jamais convergé** sur aucune fenêtre — il atteint systématiquement
  la limite de temps.
- Les logs du run overnight ne sont pas conservés (répertoires temporaires supprimés par `_cleanup()`).
- **Recommandation** : tester avec `--time_limit 1800` (30 min) pour réduire la durée totale
  (~3-5h vs ~30h) sans perte de qualité significative attendue.

**Paramètres du run overnight (référence)** :
```
python main.py --index russell3000 --cardinality 300 --solution_name quob \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --replicator_cores 8 --time_limit 10800 --distance_method pearson \
  --missing_policy strict --hard_clip 1.0
```

**Commande d'analyse correspondante** :
```
python scripts/analyze_results.py \
  --index russell3000 --cardinality 300 --solutions quob \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --missing_policy strict --hard_clip 1.0
```

---

## Améliorations potentielles identifiées

### Méthodologie — à investiguer
- **Univers de test** : actuellement on charge la composition à `start_datetime` (= `end_datetime` du training précédent). On pourrait envisager d'utiliser la composition à `end_datetime` du test aussi, pour mieux refléter les constituants réels pendant la période de détention — mais cela introduirait du look-ahead (on ne connaît pas la composition future au moment de l'investissement). La logique actuelle est donc correcte.
- **Pondération QP avec contrainte μ** : le fallback sans contrainte de rendement moyen (si SLSQP ne converge pas) devrait être loggué et suivi. Si le fallback est fréquent, envisager une formulation plus robuste (ex. relaxation de la contrainte en pénalité).
- **Convergence ReplicaTOR** : l'algorithme n'atteint jamais sa convergence naturelle (round_limit=100M) même à 10 800s. Explorer des heuristiques de warm-start ou réduire n par clustering préalable.
- **Distance dcor vs pearson** : le premier run Russell utilisait `pearson` (plus rapide). Comparer les résultats out-of-sample avec `dcor` pour valider le choix.
- **Cardinalité K** : K=300 sur Russell 3000 (~3000 stocks) = 10% de l'indice. Tester K=100, 200 pour évaluer le trade-off cardinalité / tracking error.

### Infrastructure
- **Logs persistants** : les logs ReplicaTOR sont perdus (répertoires temporaires supprimés par `_cleanup()`). Envisager de conserver un résumé (rounds exécutés, meilleure solution) par fenêtre dans le `.pkl`.
- **Licence Gurobi expirée** : la licence WLS (ID 2736279) a expiré. Gurobi n'est plus utilisable pour les runs de validation rapide — utiliser QUOB ou les solveurs lagrange.

---

## Résultats produits

| Fichier | Indice | Méthode | Cardinalité | Notes |
|---------|--------|---------|-------------|-------|
| `results/portfolio_russell3000_quob_300.pkl` | Russell 3000 | QUOB | 300 | Run overnight, time_limit=10800s, avant fix Phase 7 |
| `results/analysis_russell3000_300/` | Russell 3000 | QUOB | 300 | Analyse du run ci-dessus |
| Run en cours (4 mars 2026) | Russell 3000 | QUOB | 300 | time_limit=1800s, **avec fix Phase 7** |

---

## Variables d'environnement

| Variable | Usage |
|----------|-------|
| `REPLICATOR_PATH` | Chemin vers le binaire ReplicaTOR compilé (défaut : `~/or_tool/ReplicaTOR/cmake-build`) |
| `REPLICATOR_CORES` | Nombre de threads OpenMP pour ReplicaTOR (défaut : 8) |

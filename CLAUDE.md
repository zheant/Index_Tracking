# CLAUDE.md — Index_Tracking

## Vue d'ensemble

Réplication d'indice boursier par sous-portefeuille : sélectionner K actions parmi les constituants d'un indice (SP500, Russell 3000) et leur affecter des poids optimaux pour minimiser le tracking error hors-échantillon.

---

## Architecture

```
Index_Tracking/
├── main.py                          # Point d'entrée CLI
├── prafa/
│   ├── universe.py                  # Univers de stocks, nettoyage des données
│   ├── portfolio.py                 # Classe Portfolio + Solution (orchestre les solveurs)
│   ├── quob.py                      # Solveur QUOB (via ReplicaTOR binaire)
│   ├── gurobi.py                    # Solveur Gurobi MILP (licence expirée)
│   └── matrix_utils.py             # Matrices de distance (dcor, Pearson)
├── scripts/
│   ├── analyze_results.py           # Backtest et visualisation
│   ├── run_dscale_experiment.py     # Sweep automatisé de d_scale
│   ├── plot_regime_change.py        # Graphique changement de régime large/small caps
│   ├── build_synthetic_index.py     # Construction indice synthétique cap-pondéré (R3000, R1000)
│   ├── prepare_russell1000.py       # Préparation données Russell 1000 depuis R3000
│   ├── prepare_russell_constituents.py
│   ├── download_wrds_russell_data.py
│   └── download_mktcap_data.py
├── financial_data/
│   ├── sp500/                       # returns_stocks.csv, returns_index.csv (^SP500TR), constituants/
│   ├── russell3000/                 # returns_stocks.csv, returns_index.csv (synthétique), mktcap_stocks.csv, constituants/
│   └── russell1000/                 # idem (sous-ensemble R3000, constituants top-1000 mktcap)
└── results/                         # PKLs + analyses PNG
```

---

## Paramètres CLI (`main.py`)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--index` | `sp500` | `sp500` ou `russell3000` |
| `--solution_name` | `quob` | `quob`, `quob_stratified`, `gurobi`, `lagrange_*` |
| `--cardinality` | `50` | Nombre de stocks K |
| `--T` | `3` | Durée fenêtre d'entraînement (années) |
| `--rebalancing` | `12` | Fréquence rééquilibrage (mois) |
| `--missing_policy` | `auto` | `legacy` (SP500) ou `strict` (Russell) |
| `--reconstitution_month` | `7` | Mois d'activation de la composition Russell |
| `--max_missing_frac` | `0.10` | Fraction max NaN tolérée par stock (training) |
| `--min_trading_frac` | `0.50` | Fraction min jours à rendement non nul (liquidité) |
| `--winsor_sigma` | `3.0` | Seuil winsorisation en σ (training) |
| `--hard_clip` | `1.0` | Clip absolu ±% (training + test) |
| `--distance_method` | `dcor` | `dcor` ou `pearson` |
| `--time_limit` | `300` | Limite de temps ReplicaTOR et Gurobi (s) |
| `--d_scale` | `1.0` | D_scale_factor ReplicaTOR |
| `--strata_large_size` | `1000` | Taille de la strate large/mid cap (quob_stratified) |
| `--no_stratification` | `False` | Désactive la stratification — un seul QUOB sur Pool A entier |
| `--phase18_qp_index` | `False` | Phase 18 : pondération QP ciblant r_index |
| `--phase19_medoid_capweight` | `False` | Phase 19 : pondération par mktcap du médoïde uniquement |

---

## Indice de référence (benchmark)

### SP500
`returns_index.csv` contient `^SP500TR` — l'indice S&P 500 Total Return officiel cap-pondéré, téléchargé depuis Yahoo Finance. Données disponibles depuis **2008-01-02**.

### Russell 3000 et Russell 1000 — indice synthétique cap-pondéré
L'indice officiel FTSE Russell n'est pas accessible quotidiennement via WRDS (la librairie `ftsesamp_russell_us` ne contient qu'un échantillon mensuel de 195 stocks sur 104 dates). Le benchmark utilisé est un **indice synthétique cap-pondéré** construit depuis les données CRSP :

$$r_{\text{index}}(t) = \sum_{i \in \mathcal{C}(t)} w_i(m) \cdot r_i(t)$$

où $w_i(m) = \text{mktcap}_i(m) / \sum_j \text{mktcap}_j(m)$, $m$ étant le dernier snapshot mensuel disponible avant $t$, et $\mathcal{C}(t)$ l'ensemble des constituants actifs à la date $t$ (selon le fichier de reconstitution annuel).

**Script** : `scripts/build_synthetic_index.py`

```bash
python scripts/build_synthetic_index.py --index russell3000
python scripts/build_synthetic_index.py --index russell1000
```

**Écarts avec l'indice officiel** :
- Mktcap totale vs float-adjusted (Russell utilise le float) → faible impact
- Snapshots annuels vs mises à jour continues (IPOs, délistings mid-year) → modéré
- Pour Russell 1000 : constituants approchés (top-1000 mktcap de R3000) → très faible

**Note historique** : avant la correction, `returns_index.csv` R3000 était une **moyenne equal-weighted** des constituants (bug silencieux). Tous les runs antérieurs à la correction utilisent ce benchmark erroné.

---

## Problèmes méthodologiques identifiés

### 1. Lookback insuffisant sur les premières fenêtres (Russell uniquement)
Les données CRSP Russell ont été téléchargées à partir de **2014-01-01**, mais avec `--start_date 2014-01-02` et `T=3` ans, la première fenêtre d'entraînement couvre [2011-01-02 → 2014-01-02]. Les 6 premières fenêtres (2014–2017) ont donc des données d'entraînement insuffisantes :

| Fenêtre test | Données réelles disponibles |
|---|---|
| 2014-01 → 2014-07 | 1 jour |
| 2014-07 → 2015-01 | ~6 mois |
| 2015-01 → 2016-07 | 1–2 ans |
| 2016-07 → 2017-01 | ~2.5 ans |
| **2017-01+** | **3 ans complets ✓** |

**Correction** : re-télécharger depuis **2011-01-01** et utiliser `--start_date 2014-01-02` pour les nouveaux runs (lookback 3 ans complets dès la première fenêtre). Le SP500 n'est pas affecté (données disponibles depuis 2008).

### 2. Benchmark equal-weighted pour Russell 3000 (corrigé)
Le fichier `returns_index.csv` original était construit comme `df_returns.mean(axis=1)` — moyenne equal-weighted des constituants, pas un indice cap-pondéré. Le pipeline (ReplicaTOR + cluster cap-weighting) est conçu pour répliquer un indice cap-pondéré. **Tous les runs antérieurs mesurent le tracking contre un benchmark incorrect.**

**Correction** : `scripts/build_synthetic_index.py` génère un indice cap-pondéré cohérent.

---

## Décisions architecturales clés

### Anti-look-ahead
- **Training** : `ref_date = end_datetime` → composition à la date de décision. Winsorisation et filtres calculés sur training uniquement.
- **Test** : `ref_date = start_datetime`, `training=False` → seul hard-clip appliqué (valeur fixe).
- Fenêtre de test : `portfolio_key + 1 BDay` → `dates[i+1] - 1 BDay`.

### Politique de données manquantes
- `legacy` : fillna(0), aucun filtrage — comportement SP500 original.
- `strict` : pipeline complet (suppression lacunaires → fillna(0) → hard-clip → filtre liquidité → winsorisation → suppression NaN index).

### Constituants Russell 3000
- Fichiers `{year}.csv` sous `constituants/` : composition post-reconstitution de juin {year}, active à partir de juillet.
- `_effective_constituent_year()` gère le décalage. Reset `self.year = -1` en fin de `new_universe()`.

---

## Méthode actuelle — `quob_stratified` (Phase 12, référence)

### Architecture two-pool
- **Pool A** : stocks passant `min_trading_frac` → candidats médoïdes ReplicaTOR.
- **Pool B** : stocks passant `max_missing_frac` mais échouant `min_trading_frac` → contribuent uniquement à la pondération.

### Stratification (allocation de Neyman)
1. Diviser l'univers en strate large/mid (top `strata_large_size` par mktcap) et strate small.
2. Allouer $K_h \propto n_h \times \bar{d}_h$ médoïdes par strate.
3. Lancer ReplicaTOR indépendamment sur chaque strate (Pool A uniquement).
4. Cluster cap-weighting : poids médoïde j = Σ mktcap(cluster j, Pool A + Pool B) / mktcap_total.
5. Poids globaux = poids within-strate × fraction mktcap de la strate.

### Fallback
- Si `mktcap_weights` non disponible (SP500, pas de `mktcap_stocks.csv`) → QP SLSQP classique avec contrainte μ.
- SP500 : Pool B vide (tous les constituants sont liquides), comportement identique à l'ancien `quob`.

---

## Fonction objective de ReplicaTOR

$$C(S) = -D\_scale \sum_{\{i,j\} \subseteq S} D_{ij} + B\_scale \sum_{i \in S} \sum_{j=1}^{n} D_{ij}$$

- **Terme 1** (dispersion) : maximise la distance entre médoïdes → couverture large de l'espace.
- **Terme 2** (centralité) : préfère les stocks centraux (petite somme de ligne = proche de tous).
- `B_scale = 0.5 * (K+1) / n` — calibration recommandée, ne pas modifier.
- `D_scale` (paramètre `--d_scale`) : levier propre sur la dispersion. Ratio D_scale/B_scale contrôle la balance diversité/centralité.

**Ce n'est PAS du k-médoïdes classique.** L'assignation des stocks aux médoïdes se fait séparément dans `generateAssignments()` après la résolution.

---

## Analyse structurelle — Architecture deux pools

### Mécanisme du biais potentiel
Le découplage poids/rendements est la source principale de biais dans `quob_stratified` :
- Pool B contribue aux poids cap-weight des médoïdes mais n'est pas détenu → poids et rendements structurellement découplés.
- Le découplage est **régime-dépendant** : en régime normal (large caps > small caps), Pool B sous-performe → sur-performance chronique. En régime small-cap, Pool B surperforme → sous-performance temporaire.
- La racine du problème : les stocks Pool B **ne peuvent pas être détenus** (illiquides → cash drag en test).

### Variantes de pondération testées

**Phase 12 (référence) — Pool A + Pool B dans le cap-weighting**
- `w_j = Σ mktcap(cluster_j, Pool A ∪ Pool B) / mktcap_total`
- Formule originale avec stratification de Neyman.

**Phase 16 — Pool A cap-weighting uniquement**
- `w_j = Σ mktcap(cluster_j, Pool A) / Σ_k mktcap(cluster_k, Pool A)` — Pool B ignoré
- Résultats à réévaluer avec données corrigées (lookback 2011, indice synthétique).

**Phase 17 — Sans filtre de liquidité**
- `min_trading_frac=0.0` → Pool B vide, tous les stocks candidats médoïdes
- `w_j = Σ mktcap(cluster_j) / mktcap_total` — pool unique
- Résultats à réévaluer avec données corrigées.

**Phase 17 sans stratification**
- Pool unique, pas de stratification de Neyman — un seul QUOB sur tous les stocks.

**Phase 18 — QP ciblant r_index**
- `min ||R_med @ w - r_index||²  s.t. Σw=1, w≥0`
- Alternative sans cap-weighting — résultats à réévaluer.

**Phase 19 — Médoïde cap-weighting pur**
- `w_j = mktcap(médoïde_j) / Σ mktcap(médoïdes)`
- Résultats à réévaluer avec données corrigées.

### Le Russell 3000 : considérations structurelles
| Indice | Réplicabilité | Raison |
|---|---|---|
| SP500 | Excellente | Tous constituants liquides, Pool B vide |
| Russell 1000 | Bonne | Large/mid caps, liquidité acceptable |
| Russell 3000 | Limitée structurellement | ~15-20% du poids dans des illiquides non-réplicables |
| Russell 2000 | Très difficile | Quasi-entièrement Pool B |

### Reproductibilité
- **Méthodologique** : oui — framework transposable à tout indice.
- **Numérique** : non — ReplicaTOR parallel tempering sans seed fixé, résultats varient entre runs et selon le matériel.

---

## Conclusions empiriques — Sweep d_scale Phase 17 (Russell 3000, K=300, 2014–2023)

> Runs effectués avec données corrigées (lookback 2011, indice synthétique cap-pondéré), `--no_stratification --min_trading_frac 0.0`.

### Tableau de synthèse

| Méthode | Cum. Rp | Cum. Indice | Écart | TE ann. | Biais ann. | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|
| Phase 12 (2 pools + strat, mtf=0.20) | 135.3% | 215.0% | -79.7pp | 8.37% | -2.35% | +0.60 | 12.7 |
| **Ph17 d=0.5** (référence nouvelle) | **199.5%** | 215.0% | **-15.5pp** | **4.50%** | **-0.24%** | +0.40 | 6.7 |
| Ph17 d=1.0 | 152.1% | 215.0% | -62.9pp | 8.22% | -1.82% | -0.19 | 10.9 |
| Ph17 d=2.0 | 154.6% | 215.0% | -60.5pp | 9.17% | -1.86% | -2.11 | 42.6 |
| Ph17 d=5.0 | 178.5% | 215.0% | -36.5pp | 8.84% | -1.02% | +0.73 | 12.4 |
| Ph17 d=10.0 | 532.2% | 215.0% | +317pp | 8.61% | +7.15% | +0.63 | 8.1 |

### Conclusions clés

**1. Phase 12 est contre-productive.**
La stratification de Neyman + Pool B dans le cap-weighting génère un biais négatif structurel de **-4.06%/yr pré-2020**. Le retournement à +0.20%/yr post-2020 confirme l'hypothèse régime-dépendante : Pool B (illiquides) sous-performe en régime large-cap, surperforme en régime small-cap (rally H2 2020–2021). La complexité de Phase 12 joue contre la performance.

**2. Ph17 d=0.5 est la configuration optimale pour Russell 3000.**
TE annualisée de **4.50%** — quasi 2× mieux que toutes les autres configurations. Biais quasi-nul et homogène (-0.24%/yr sur toute la période). Résistance au stress COVID (TE daily 0.47% vs 1.20% pour Phase 12 sur la même fenêtre).

**3. Le terme de centralité est un mécanisme de stratification implicite.**
À d=0.5, la centralité domine : ReplicaTOR sélectionne les stocks à petite somme de ligne dans la matrice Pearson = stocks corrélés à tout l'univers = large caps. Le problème d'hétérogénéité taille/poids (large caps ~75% du poids, ~23% des stocks) est résolu sans stratification explicite. La stratification de Neyman devient redondante et son coût (couplage Pool B) l'emporte sur son bénéfice.

**4. Impact de d_scale est non-linéaire avec saut brutal entre d=0.5 et d=1.**
- d=0.5 → centralité dominante → médoïdes = large caps → TE=4.5%
- d≥1 → dispersion croissante → médoïdes incluent des small caps → TE~8–9%
- d=10 → dispersion quasi-pure → sélection de small caps extrêmes → dérive vers stratégie active non intentionnelle (+317pp vs indice)

**5. Gap résiduel de -15.5pp sur 10 ans (~1.5pp/an) avec d=0.5.**
Probablement structurel : limite de réplicabilité du Russell 3000 liée au cash drag sur délistings et aux illiquides non détenus. La TE post-2020 (~5.8%) > pré-2020 (~3.4%) confirme la dégradation liée aux délistings 2020–2023.

**6. Architecture recommandée pour Russell 3000 : pool unique, pas de stratification, d=0.5.**
Commande de référence mise à jour — voir section "Run Russell 3000 — nouvelle référence Ph17 d=0.5" ci-dessous.

---

## Résultats produits

⚠️ **Tous les runs ci-dessous utilisent un benchmark equal-weighted incorrect. Résultats à invalider — à refaire avec données corrigées (lookback 2011, indice synthétique cap-pondéré).**

| Fichier PKL | Méthode | Notes |
|-------------|---------|-------|
| `portfolio_russell3000_quob_300.pkl` | QUOB | Run overnight 10800s, avant Phase 7 — benchmark incorrect |
| `portfolio_russell3000_quob_300_rebal12_mtf050.pkl` | QUOB | rebal=12, mtf=0.50, Phase 7 — benchmark incorrect |
| `portfolio_russell3000_quob_stratified_300_phase12_capweight.pkl` | QUOB stratifié Ph.12 | Référence — benchmark incorrect |
| `portfolio_russell3000_quob_stratified_300.pkl` | QUOB stratifié Ph.14 | QP ciblé r_strate — benchmark incorrect |
| `portfolio_russell3000_quob_stratified_300_dscale_{X}.pkl` | QUOB stratifié Ph.15 | Sweep d_scale ∈ {0.5,1,2,5,10} — benchmark incorrect |
| `portfolio_russell3000_quob_stratified_300_phase16_pool_a_only.pkl` | QUOB stratifié Ph.16 | Pool A cap-weight uniquement — benchmark incorrect |
| `portfolio_russell3000_quob_stratified_300_phase17_no_liq_filter.pkl` | QUOB stratifié Ph.17 | Sans filtre liquidité — benchmark incorrect |
| `portfolio_russell3000_quob_stratified_300_phase17_no_strat_dscale_{X}.pkl` | QUOB Ph.17 no-strat | Sweep d_scale, no_stratification — benchmark incorrect |
| `portfolio_russell3000_quob_stratified_300_phase18_qp_index.pkl` | QUOB stratifié Ph.18 | QP sur r_index — benchmark incorrect |
| `portfolio_russell3000_quob_stratified_300_phase19_medoid_capweight.pkl` | QUOB stratifié Ph.19 | Médoïde cap-weight pur — benchmark incorrect |

**Runs avec données corrigées (lookback 2011, indice synthétique cap-pondéré) :**

| Fichier PKL | Méthode | Résultats |
|-------------|---------|-----------|
| `portfolio_russell3000_quob_stratified_300.pkl` | Phase 12 (2 pools + strat, mtf=0.20) | Cum=135.3%, TE=8.37%, biais=-2.35%/yr — **invalidé** |
| `portfolio_russell3000_quob_stratified_300_phase17_no_strat_dscale_0.5.pkl` | Ph17 no-strat d=0.5 | Cum=199.5%, TE=4.50%, biais=-0.24%/yr — **référence** |
| `portfolio_russell3000_quob_stratified_300_phase17_no_strat_dscale_1.pkl` | Ph17 no-strat d=1.0 | Cum=152.1%, TE=8.22%, biais=-1.82%/yr |
| `portfolio_russell3000_quob_stratified_300_phase17_no_strat_dscale_2.pkl` | Ph17 no-strat d=2.0 | Cum=154.6%, TE=9.17%, kurtosis=42.6 (anomalie) |
| `portfolio_russell3000_quob_stratified_300_phase17_no_strat_dscale_5.pkl` | Ph17 no-strat d=5.0 | Cum=178.5%, TE=8.84%, biais=-1.02%/yr |
| `portfolio_russell3000_quob_stratified_300_phase17_no_strat_dscale_10.pkl` | Ph17 no-strat d=10.0 | Cum=532.2% — dérive active, inutilisable |

---

## Commandes de référence

### Run Russell 3000 — nouvelle référence Ph17 d=0.5 (pool unique, pas de stratification)
```bash
# Nécessite données téléchargées depuis 2011-01-01
python main.py --index russell3000 --cardinality 300 --solution_name quob_stratified \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --rebalancing 6 --min_trading_frac 0.0 \
  --replicator_cores 128 --time_limit 112 \
  --distance_method pearson --missing_policy strict --hard_clip 1.0 \
  --no_stratification --d_scale 0.5
```

### Run Russell 3000 — Phase 12 (deux pools + stratification, ancienne référence)
```bash
# Nécessite données téléchargées depuis 2011-01-01
python main.py --index russell3000 --cardinality 300 --solution_name quob_stratified \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --rebalancing 6 --min_trading_frac 0.20 \
  --replicator_cores 128 --time_limit 112 \
  --distance_method pearson --missing_policy strict --hard_clip 1.0
```

### Phase 16 — Pool A cap-weighting uniquement
```bash
# Nécessite données téléchargées depuis 2011-01-01
python main.py --index russell3000 --cardinality 300 --solution_name quob_stratified \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --rebalancing 6 --min_trading_frac 0.20 \
  --replicator_cores 128 --time_limit 112 \
  --distance_method pearson --missing_policy strict --hard_clip 1.0 \
  --exclude_pool_b_capweight
```

### Phase 17 — Sans filtre de liquidité
```bash
# Nécessite données téléchargées depuis 2011-01-01
python main.py --index russell3000 --cardinality 300 --solution_name quob_stratified \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --rebalancing 6 --min_trading_frac 0.0 \
  --replicator_cores 128 --time_limit 112 \
  --distance_method pearson --missing_policy strict --hard_clip 1.0
```

### Phase 17 — Sans stratification
```bash
# Nécessite données téléchargées depuis 2011-01-01
python main.py --index russell3000 --cardinality 300 --solution_name quob_stratified \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --rebalancing 6 --min_trading_frac 0.0 \
  --replicator_cores 128 --time_limit 112 \
  --distance_method pearson --missing_policy strict --hard_clip 1.0 \
  --no_stratification
```

### Phase 18 — QP ciblant r_index
```bash
# Nécessite données téléchargées depuis 2011-01-01
python main.py --index russell3000 --cardinality 300 --solution_name quob_stratified \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --rebalancing 6 --min_trading_frac 0.20 \
  --replicator_cores 128 --time_limit 112 \
  --distance_method pearson --missing_policy strict --hard_clip 1.0 \
  --phase18_qp_index
```

### Phase 19 — Médoïde cap-weighting pur
```bash
# Nécessite données téléchargées depuis 2011-01-01
python main.py --index russell3000 --cardinality 300 --solution_name quob_stratified \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --rebalancing 6 --min_trading_frac 0.20 \
  --replicator_cores 128 --time_limit 112 \
  --distance_method pearson --missing_policy strict --hard_clip 1.0 \
  --phase19_medoid_capweight
```

### Analyse Russell 3000
```bash
python scripts/analyze_results.py \
  --index russell3000 --cardinality 300 --solutions quob_stratified \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --rebalancing 6 --min_trading_frac 0.20 \
  --missing_policy strict --hard_clip 1.0
```

### Run Russell 1000
```bash
python main.py --index russell1000 --cardinality 100 \
  --solution_name quob_stratified \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --rebalancing 6 --min_trading_frac 0.20 \
  --replicator_cores 128 --time_limit 112 \
  --distance_method pearson --missing_policy strict --hard_clip 1.0

### Analyse Russell 1000
python scripts/analyze_results.py \
  --index russell1000 --cardinality 100 --solutions quob_stratified \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --rebalancing 6 --min_trading_frac 0.20 \
  --missing_policy strict --hard_clip 1.0
```

### Sweep d_scale — baseline (Phase 17 no-strat, un seul pool, pas de filtre liquidité)
```bash
# Nécessite données depuis 2011-01-01. Supprimer les anciens PKLs si présents :
# rm results/portfolio_russell3000_quob_stratified_300_phase17_no_strat_dscale_*.pkl
python scripts/run_dscale_experiment.py \
    --d_scales 0.5 1.0 2.0 5.0 10.0 \
    --start_date 2014-01-02 \
    --min_trading_frac 0.0 --no_stratification \
    --replicator_cores 128 --time_limit 112 \
    --output_dir results/dscale_experiment_phase17_no_strat
# Analyse seule (PKLs déjà présents) :
python scripts/run_dscale_experiment.py \
    --d_scales 0.5 1.0 2.0 5.0 10.0 \
    --start_date 2014-01-02 \
    --min_trading_frac 0.0 --no_stratification \
    --output_dir results/dscale_experiment_phase17_no_strat --skip_runs
```

### Sweep d_scale — Phase 12 référence (deux pools, stratification)
```bash
# Nécessite données depuis 2011-01-01. Supprimer les anciens PKLs si présents :
# rm results/portfolio_russell3000_quob_stratified_300_dscale_*.pkl
python scripts/run_dscale_experiment.py \
    --d_scales 0.5 1.0 2.0 5.0 10.0 \
    --start_date 2014-01-02 \
    --replicator_cores 128 --time_limit 112
# Analyse seule (PKLs déjà présents) :
python scripts/run_dscale_experiment.py \
    --d_scales 0.5 1.0 2.0 5.0 10.0 \
    --start_date 2014-01-02 --skip_runs
```

### Graphique changement de régime
```bash
python scripts/plot_regime_change.py
# Paramètres optionnels :
python scripts/plot_regime_change.py --large_n 1000 --output results/regime_change.png
```

### Téléchargement et préparation données Russell 3000 (avec lookback correct)
```bash
# Returns depuis 2011 (lookback 3 ans pour start_date 2014-01-02)
python scripts/download_wrds_russell_data.py \
    --permno-csv financial_data/russell3000/constituants/all_permnos.csv \
    --start-date 2011-01-01 --end-date 2023-12-31 \
    --skip-index --output-dir financial_data/russell3000

# Market cap depuis 2011
python scripts/download_mktcap_data.py \
    --permno-csv financial_data/russell3000/constituants/all_permnos.csv \
    --start-date 2011-01-01 --end-date 2023-12-31 \
    --output-dir financial_data/russell3000

# Indice synthétique cap-pondéré
python scripts/build_synthetic_index.py --index russell3000
```

### Préparation données Russell 1000
```bash
# Étape 1 — Constituants + filtrage returns/mktcap depuis R3000 (offline)
python scripts/prepare_russell1000.py \
    --r3000-dir financial_data/russell3000 \
    --output-dir financial_data/russell1000 --size 1000

# Étape 2 — Indice synthétique cap-pondéré
python scripts/build_synthetic_index.py --index russell1000
```

---

## Variables d'environnement

| Variable | Usage |
|----------|-------|
| `REPLICATOR_PATH` | Chemin vers le binaire ReplicaTOR (défaut : `~/or_tool/ReplicaTOR/cmake-build`) |
| `REPLICATOR_CORES` | Nombre de threads OpenMP (défaut : 8) |

---

## Améliorations potentielles

- ~~**Normalisation des températures ReplicaTOR par d_scale**~~ **Testé (Phase 20) puis rollback** : `T_max/T_min` proportionnels à `d_scale` testés mais résultats détériorés — températures revenues aux valeurs fixes `0.01 / 0.00001`.
- ~~**Russell 1000 comme cible**~~ **En cours** : pipeline fonctionnel, données préparées, runs à relancer avec données corrigées (lookback 2011, indice synthétique cap-pondéré).
- ~~**Refaire tous les runs R3000**~~ **Fait** : sweep d_scale Ph17 no-strat + Phase 12 refaits avec données corrigées. Nouvelle référence : Ph17 d=0.5 (voir section Conclusions empiriques).
- **Tester d_scale ∈ {0.1, 0.2, 0.3}** sur Russell 3000 : la relation TE/d_scale est non-linéaire avec un saut brutal entre 0.5 et 1.0 — vérifier si d < 0.5 continue d'améliorer ou si 0.5 est un minimum.
- **Cash drag sur délistings** : redistribuer dynamiquement les poids des stocks délistés vers les actifs en test — probablement le levier le plus impactant pour réduire le gap post-2020 sur Russell.
- **Distance Pearson sur jours actifs uniquement** : calculer la corrélation uniquement sur les jours où les deux stocks tradent (`r_i ≠ 0` AND `r_j ≠ 0`) — élimine le gonflement artificiel de corrélation entre illiquides. Modification dans `matrix_utils.py`.
- **EWMA des rendements** : pondération exponentielle `r̃_{i,t} = λ^{(T-t)/2} × r_{i,t}` avant calcul de distance et QP — meilleure alternative à une fenêtre courte.
- **Cardinalité K** : tester K=100, 200 sur Russell 3000 (actuellement K=300 = 10% de l'indice).
- **Logs persistants** : les logs ReplicaTOR sont perdus (répertoires temporaires supprimés). Envisager un résumé (rounds, meilleure solution) sauvegardé dans le PKL.
- **Licence Gurobi** : WLS ID 2736279 expirée. Gurobi inutilisable — utiliser QUOB ou lagrange.

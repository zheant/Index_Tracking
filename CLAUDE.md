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
│   ├── prepare_russell_constituents.py
│   ├── download_wrds_russell_data.py
│   └── download_mktcap_data.py
├── financial_data/
│   ├── sp500/                       # returns_stocks.csv, returns_index.csv, constituants/
│   └── russell3000/                 # idem + mktcap_stocks.csv
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

## Expérience d_scale — Résultats (Phase 15, 6 mars 2026)

Run : `quob_stratified`, K=300, rebal=6, mtf=0.20, pearson, strict, hard_clip=1.0, 64 cœurs.

| d_scale | Biais (/j) | ~Biais annuel | Variance | TE moyen | Fin 2023 (×) |
|---------|-----------|--------------|----------|----------|------------|
| 0.5 | +0.000165 | +4.2%/an | 0.000065 | 0.007 | ×4.3 |
| **1.0** | **+0.000080** | **+2.0%/an** | **0.000048** | **0.0065** | **×3.5** |
| 2.0 | +0.000040 | +1.0%/an | 0.000070 | 0.0073 | ×2.5 |
| 5.0 | −0.000025 | −0.6%/an | 0.000100 | 0.0075 | ×2.0 ❌ |
| 10.0 | +0.000100 | +2.5%/an | 0.000053 | 0.0068 | ×3.5 |

*Indice Russell 3000 : ×2.8 fin 2023.*

**Conclusions** :
- La relation biais/d_scale est **non-monotone** : décroît jusqu'à d_scale=5 puis remonte.
- d_scale=5 : seul run sous-performant l'indice, variance ×2 — médoïdes trop dispersés, clusters incohérents.
- d_scale=1 : meilleur équilibre (TE minimal, variance faible, sur-performance économiquement justifiable).
- **d_scale comme levier de contrôle du biais est épuisé** : effets secondaires > bénéfice dès d_scale ≥ 2.

---

## Analyse du biais structurel Russell 3000

### Cause de la sur-performance post-2021
- **Mécanisme** : le portefeuille détient les rendements de Pool A (liquides) mais porte les poids market-cap de Pool A + Pool B (incluant illiquides). En régime quality/large-cap (2022-2023), Pool B sous-performe fortement → le portefeuille encaisse les poids mais pas les pertes.
- **2020** : le rally small-cap (Russell 2000 ×2 en 6 mois) a temporairement inversé la prime de liquidité — biais masqué. Post-2021, le régime habituel (qualité > illiquidité) se rétablit et amplifie la sur-performance.

### Origine du biais : architecture deux pools

Le biais de liquidité n'est pas une conséquence indirecte — il est **directement causé** par la séparation en deux pools :
- Pool B contribue aux poids cap-weight des médoïdes mais n'est pas détenu → les poids et rendements sont structurellement découplés.
- Sans cette séparation (pure Pool A cap-weighting), le biais disparaît — au coût d'ignorer ~15-20% du poids de l'indice.
- Le découplage est **régime-dépendant** : en régime normal (large caps > small caps), Pool B sous-performe → sur-performance chronique. En régime small-cap (2020), Pool B surperforme → sous-performance temporaire.

### Pourquoi les corrections tentées ont échoué
- **Phase 13/14** — QP ciblé sur `r_strate(Pool A+B)` : biais −1.8%/an car `μ(A+B) < μ(A)` structurellement.
- **d_scale=5** — dispersion forcée : sous-performance + variance ×2.
- La racine du problème : les stocks Pool B **ne peuvent pas être détenus** (illiquides → cash drag en test). Tout système forçant le portefeuille à cibler leur rendement impose des poids négatifs implicites sur les actifs performants.

### Pistes non encore testées (Phase 16+)

**Piste 1 — Pool A cap-weighting uniquement (Phase 16)** *(priorité)*
- Exclure Pool B du cap-weighting : `poids_médoïde_j = Σ mktcap(cluster_j, Pool A) / mktcap_Pool_A_total`
- Renormalisation sur Pool A → somme des poids = 1, Pool B ignoré économiquement
- `min_trading_frac = 0.20` conservé → univers réduit à ~80-85% du Russell 3000
- Élimine le biais de liquidité et la sensibilité aux changements de régime
- Contrepartie : on réplique un "Russell 3000 filtré par liquidité", pas l'indice exact

**Piste 2 — Sans filtre de liquidité (Phase 17)**
- Supprimer `min_trading_frac` → tous les constituants sont candidats médoïdes, pas de séparation Pool A/B
- Réplication fidèle de l'indice sur le papier, sans biais de liquidité
- Limitation pratique : médoïdes illiquides inexécutables pour portefeuilles institutionnels (>10M$) — acceptable en contexte académique
- À tester après Phase 16

### Le Russell 3000 est un mauvais candidat pour cette méthode
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

## Résultats produits

| Fichier PKL | Méthode | Notes |
|-------------|---------|-------|
| `portfolio_russell3000_quob_300.pkl` | QUOB | Run overnight 10800s, avant Phase 7 |
| `portfolio_russell3000_quob_300_rebal12_mtf050.pkl` | QUOB | rebal=12, mtf=0.50, Phase 7 |
| `portfolio_russell3000_quob_stratified_300_phase12_capweight.pkl` | QUOB stratifié Ph.12 | **Référence — biais +0.00004/j** |
| `portfolio_russell3000_quob_stratified_300.pkl` | QUOB stratifié Ph.14 | QP ciblé r_strate, biais −0.00007/j |
| `portfolio_russell3000_quob_stratified_300_dscale_{X}.pkl` | QUOB stratifié Ph.15 | Sweep d_scale ∈ {0.5,1,2,5,10} |

---

## Commandes de référence

### Run Russell 3000 (paramètres actuels, 128 cœurs)
```bash
python main.py --index russell3000 --cardinality 300 --solution_name quob_stratified \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --rebalancing 6 --min_trading_frac 0.20 \
  --replicator_cores 128 --time_limit 112 \
  --distance_method pearson --missing_policy strict --hard_clip 1.0
```

### Phase 16 — Pool A cap-weighting uniquement (à implémenter)
```bash
# Même commande, cap-weighting modifié dans portfolio.py pour exclure Pool B
python main.py --index russell3000 --cardinality 300 --solution_name quob_stratified \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --rebalancing 6 --min_trading_frac 0.20 \
  --replicator_cores 128 --time_limit 112 \
  --distance_method pearson --missing_policy strict --hard_clip 1.0
```

### Phase 17 — Sans filtre de liquidité (à implémenter)
```bash
# min_trading_frac=0.0 → pas de Pool B, tous les stocks sont candidats médoïdes
python main.py --index russell3000 --cardinality 300 --solution_name quob_stratified \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --rebalancing 6 --min_trading_frac 0.0 \
  --replicator_cores 128 --time_limit 112 \
  --distance_method pearson --missing_policy strict --hard_clip 1.0
```

### Analyse Russell 3000
```bash
python scripts/analyze_results.py \
  --index russell3000 --cardinality 300 --solutions quob_stratified \
  --start_date 2014-01-02 --end_date 2023-12-31 \
  --rebalancing 6 --min_trading_frac 0.20 \
  --missing_policy strict --hard_clip 1.0
```

### Sweep d_scale
```bash
python scripts/run_dscale_experiment.py \
    --d_scales 0.5 1.0 2.0 5.0 10.0 \
    --replicator_cores 128 --time_limit 112
# Analyse seule (PKLs déjà présents) :
python scripts/run_dscale_experiment.py --d_scales 0.5 1.0 2.0 5.0 10.0 --skip_runs
```

### Téléchargement market cap (si à refaire)
```bash
python scripts/download_mktcap_data.py \
    --permno-csv financial_data/russell3000/constituants/all_permnos.csv \
    --start-date 2014-01-01 --end-date 2023-12-31
```

---

## Variables d'environnement

| Variable | Usage |
|----------|-------|
| `REPLICATOR_PATH` | Chemin vers le binaire ReplicaTOR (défaut : `~/or_tool/ReplicaTOR/cmake-build`) |
| `REPLICATOR_CORES` | Nombre de threads OpenMP (défaut : 8) |

---

## Améliorations potentielles

- **Cash drag sur délistings** : redistribuer dynamiquement les poids des stocks délistés vers les actifs en test — probablement le levier le plus impactant pour réduire le gap post-2020 sur Russell.
- **Distance Pearson sur jours actifs uniquement** : calculer la corrélation uniquement sur les jours où les deux stocks tradent (`r_i ≠ 0` AND `r_j ≠ 0`) — élimine le gonflement artificiel de corrélation entre illiquides. Modification dans `matrix_utils.py`.
- **EWMA des rendements** : pondération exponentielle `r̃_{i,t} = λ^{(T-t)/2} × r_{i,t}` avant calcul de distance et QP — meilleure alternative à une fenêtre courte.
- **Cardinalité K** : tester K=100, 200 sur Russell 3000 (actuellement K=300 = 10% de l'indice).
- **Russell 1000 comme cible** : universiplus homogène, réplication structurellement plus propre.
- **Logs persistants** : les logs ReplicaTOR sont perdus (répertoires temporaires supprimés). Envisager un résumé (rounds, meilleure solution) sauvegardé dans le PKL.
- **Licence Gurobi** : WLS ID 2736279 expirée. Gurobi inutilisable — utiliser QUOB ou lagrange.

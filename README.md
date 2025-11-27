This repository contains my research project completed during my Summer 2025 internship at CIRRELT.

👉 [Internship Report (PDF)](https://github.com/aubejay22/Index_Tracking/raw/main/Recherche_ete25%20(4).pdf)

## Running the Russell 3000 workflow

### Python environment setup

Create an isolated environment in the repo root and install the dependencies:

```bash
cd ~/Index_Tracking
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

When you open a new shell, re-activate the environment with `source .venv/bin/activate`.

The Russell 3000 pipeline mirrors the SP500 process but uses 300 exemplars. The
commands below assume the repository is located at `~/Index_Tracking` and
ReplicaTOR is available at `~/or_tool/ReplicaTOR/cmake-build` (override with
the `REPLICATOR_PATH` environment variable if needed; if you pass a directory,
the tool will look for a `ReplicaTOR` binary inside it). Ensure the binary is
executable (e.g., `chmod +x ~/or_tool/ReplicaTOR/cmake-build/ReplicaTOR`).

1. **Prepare constituent files** (normalises the raw permno lists and creates a
   union file used for WRDS downloads):

   ```bash
   cd ~/Index_Tracking
   python scripts/prepare_russell_constituents.py
   ```

2. **Download CRSP returns from WRDS** (requires the `wrds` and
   `psycopg2-binary` packages and valid WRDS credentials/VPN):

   ```bash
   cd ~/Index_Tracking
   python scripts/download_wrds_russell_data.py \
       --permno-csv financial_data/russell3000/constituants/all_permnos.csv \
       --start-date 2014-01-01 \
       --end-date 2023-12-31
   ```

3. **Run the optimisers** (example dates match the WRDS download window). Use
   `--time_limit` to override the default 300-second cap applied to both
   ReplicaTOR and Gurobi, and `--distance_method` to choose the distance matrix
   (`dcor` or `pearson`; default is distance correlation):

   ```bash
   cd ~/Index_Tracking

   # K-medoids/ReplicaTOR + local optimisation
   # --replicator_cores (or REPLICATOR_CORES env) controls OpenMP threads (8 cores on c6i.2xlarge)
   python main.py --index russell3000 --cardinality 300 --solution_name quob \
       --start_date 2014-01-02 --end_date 2023-12-31 --result_path results \
       --replicator_cores 8 --time_limit 300 --distance_method dcor

   # Gurobi baseline
   python main.py --index russell3000 --cardinality 300 --solution_name gurobi \
       --start_date 2014-01-02 --end_date 2023-12-31 --result_path results \
       --time_limit 300 --distance_method dcor
   ```

The SP500 experiments continue to run as before; pass `--index sp500` and, if
desired, override the cardinality (default: 50) to the original 50-stock
portfolio used in the internship report.

### Russell 3000 results analysis (matches `analyses_resultats.ipynb`)

Once the optimisers have written their portfolio pickles (e.g.,
`results/portfolio_russell3000_quob_300.json` and
`results/portfolio_russell3000_gurobi_300.json`), you can recreate the
notebook-style charts for Russell 3000 with:

```bash
cd ~/Index_Tracking
python scripts/analyze_results.py \
    --index russell3000 --cardinality 300 --result_path results \
    --solutions quob gurobi --start_date 2014-01-02 --end_date 2023-12-31
```

Plots are saved under `results/analysis_russell3000_300` by default. List
additional solution names (e.g., `quob_cor`) in `--solutions` to include their
portfolios in the analysis.

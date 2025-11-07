# Russell 2000 staging helpers

This repository contains utility code that helps adapt the original S&P 500 index-tracking workflow to the Russell 2000 universe.

## Preparing constituent lists

Historical Russell 2000 composition files (one CSV per year) should live under the
`composition historique russel2000/` directory. Each CSV must contain a single
column named `permno` with the identifiers that belong to the index for that year.

Run the helper script to normalise those files into the structure expected by the
pipeline and to generate the union of all observed identifiers:

```bash
source .venv/bin/activate
python scripts/prepare_russell_constituents.py \
  --source "composition historique russel2000" \
  --output "financial_data/russel2000/constituants"
```

The script copies each yearly CSV into the output directory (sorting and
de-duplicating the permnos) and writes `all_permnos.csv` which aggregates every
unique identifier across the available years. Use that file when querying WRDS for
adjusted price histories.

If you keep the raw composition files elsewhere, override the `--source` option.
Likewise, change `--output` if you want the normalised files to be written to a
different location. Use `--union-name` to override the name of the aggregated
CSV if you need multiple universes side by side.

## Quick command recap

Copy/paste the following block on the EC2 instance (with the repository cloned)
to stage the Russell 2000 constituent universe and review the generated files:

```bash
cd ~/Index_Tracking
source .venv/bin/activate
python scripts/prepare_russell_constituents.py \
  --source "composition historique russel2000" \
  --output "financial_data/russel2000/constituants"

ls financial_data/russel2000/constituants
head -n 5 financial_data/russel2000/constituants/all_permnos.csv
```

The `ls` and `head` commands confirm that the per-year CSVs and the aggregated
`all_permnos.csv` file have been produced successfully.

## Downloading adjusted returns from WRDS

Once the constituent snapshots are ready, download the adjusted CRSP daily
returns for every permno and generate the input matrices expected by the
portfolio optimiser.

Install the WRDS client in the virtual environment if it is not already
available:

```bash
pip install wrds psycopg2-binary
```

Then run the helper script (change the dates if you need a different window):

```bash
cd ~/Index_Tracking
source .venv/bin/activate
python scripts/download_wrds_russell_data.py \
  --permno-csv financial_data/russel2000/constituants/all_permnos.csv \
  --start-date 2014-01-01 \
  --end-date 2023-12-31
```

You will be prompted for your WRDS credentials. The script stores two files
under `financial_data/russel2000/`:

* `returns_stocks.csv` — wide matrix of daily returns with one column per
  permno.
* `returns_index.csv` — an equal-weight benchmark constructed from the same
  universe (serves as a proxy index if you do not have the official Russell
  2000 total-return series).

If you see an authentication error (for example *“PAM authentication failed”*),
double-check your username/password, make sure any required institutional VPN is
connected, and confirm with WRDS support that your account is enabled for
PostgreSQL access. The script will exit early in that situation so you can retry
after fixing the connection.

Re-run the script whenever you add more constituent files or need to refresh
the data window. Pass `--skip-index` if you plan to supply a different index
return series manually.

## Troubleshooting large diffs on GitHub

If GitHub refuses to display a diff and reports that the generated diff exceeds
its size limit, the commit usually contains heavy artifacts such as the
`gurobi1200/` solver bundle or raw CSV exports under `financial_data/`. Check
[`docs/troubleshooting_diff_limit.md`](docs/troubleshooting_diff_limit.md) for
practical ways to keep those assets out of your commits or to move them to Git
LFS before opening a pull request.

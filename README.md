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

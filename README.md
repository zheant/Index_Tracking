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
different location.

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

## Choosing an EC2 instance type for ReplicaTOR

The Russell 2000 workflow is CPU bound: building distance matrices is
`O(n^2)` and ReplicaTOR spends most of its time in simulated annealing.
For good turnaround times:

* Use a compute-optimised family (for example `c6i`) so you get high
  per-core performance at a reasonable cost.
* `c6i.2xlarge` (8 vCPUs, 16 GiB RAM) is a solid baseline for 30–300
  medoids. Move to `c6i.4xlarge` or similar if you routinely run with
  higher cardinalities or longer ReplicaTOR time limits.
* Stop the instance and resize it in the AWS console before launching
  the larger jobs. Keep an eye on CPU utilisation with CloudWatch or
  `htop`; if a run consistently pegs all cores for the entire solve
  window, scale up again.
* Memory pressure is usually modest (a 3 000 × 3 000 distance matrix is
  roughly 70 MB), so you only need a memory-optimised family if you plan
  to work with substantially larger universes.

## Configuring the ReplicaTOR solver

The QUOB optimiser relies on the standalone C++ `ReplicaTOR` binary. Build it
separately (for instance under `~/or_tool/ReplicaTOR`) and keep the resulting
executable accessible on the EC2 instance. The Python wrapper will look for the
binary in the following order:

1. The `--replicator-bin` CLI flag passed to `main.py`.
2. The `REPLICATOR_BIN` environment variable.
3. Common build folders:
   - `~/or_tool/cmake-build/ReplicaTOR`
   - `~/or_tool/ReplicaTOR/cmake-build/ReplicaTOR`
   - `<repo>/or_tool/cmake-build/ReplicaTOR`
   - `<repo>/or_tool/ReplicaTOR/cmake-build/ReplicaTOR`

Example command line:

```bash
python main.py \
  --index russel2000 \
  --solution_name quob \
  --cardinality 30 \
  --start_date 2014-01-02 \
  --end_date 2023-12-31 \
  --replicator_bin /home/ubuntu/or_tool/ReplicaTOR/cmake-build/ReplicaTOR \
  --replicator_time_limit 120 \
  --replicator_cores 2
```

Adjust the arguments to match your rebalancing schedule and data paths. The same
flags work with the correlation-based variant (`quob_cor`). The optional
`--replicator_time_limit` flag lets you shorten ReplicaTOR runs (in seconds) if
the default 300 s budget per rebalance window is too long for experimentation.
`--replicator_cores` controls the `num_cores_per_controller` value written to the
ReplicaTOR parameter file. The solver expects a strictly positive power of two
and will launch that many worker threads per controller when the host has spare
vCPUs. When experimenting with larger portfolios (for example `--cardinality
300`), consider extending the time limit, increasing the core count and
upgrading the EC2 instance size so that ReplicaTOR has enough CPU headroom to
search the much larger solution space.

When ReplicaTOR finishes, the wrapper now saves the medoid assignments to
`prafa/dist_matrix/dist_matrix.clusters.txt`. The `K` medoid indices still live
in `dist_matrix.soln.txt` (or, on older builds, `dist_matrix.soln`); the wrapper
accepts both names. The companion clusters file is derived from the distance
matrix after the solve and is therefore free of the all-zero placeholder array
printed by ReplicaTOR itself. If you set `--replicator_cores` to a power of two
greater than one, the generated `.params` file reflects that choice so ReplicaTOR
can spawn additional worker threads (subject to the compiled binary supporting
multi-threading on the target instance).

## Troubleshooting large diffs on GitHub

If GitHub refuses to display a diff and reports that the generated diff exceeds
its size limit, the commit usually contains heavy artifacts such as the
`gurobi1200/` solver bundle or raw CSV exports under `financial_data/`. Check
[`docs/troubleshooting_diff_limit.md`](docs/troubleshooting_diff_limit.md) for
practical ways to keep those assets out of your commits or to move them to Git
LFS before opening a pull request.

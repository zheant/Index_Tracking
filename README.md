# Russell 3000 staging helpers

This repository contains utility code that helps adapt the original S&P 500 index-tracking workflow to the Russell 3000 universe.

## Preparing constituent lists

Historical Russell 3000 composition files (one CSV per year) should live under the
`composition historique russel3000/` directory. Each CSV must contain a single
column named `permno` with the identifiers that belong to the index for that year.

Run the helper script to normalise those files into the structure expected by the
pipeline and to generate the union of all observed identifiers:

```bash
source .venv/bin/activate
python scripts/prepare_russell_constituents.py \
  --source "composition historique russel3000" \
  --output "financial_data/russel3000/constituants"
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
to stage the Russell 3000 constituent universe and review the generated files:

```bash
cd ~/Index_Tracking
source .venv/bin/activate
python scripts/prepare_russell_constituents.py \
  --source "composition historique russel3000" \
  --output "financial_data/russel3000/constituants"

ls financial_data/russel3000/constituants
head -n 5 financial_data/russel3000/constituants/all_permnos.csv
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
  --permno-csv financial_data/russel3000/constituants/all_permnos.csv \
  --start-date 2014-01-01 \
  --end-date 2023-12-31
```

You will be prompted for your WRDS credentials. The script stores two files
under `financial_data/russel3000/`:

* `returns_stocks.csv` — wide matrix of daily returns with one column per
  permno.
* `returns_index.csv` — an equal-weight benchmark constructed from the same
  universe (serves as a proxy index if you do not have the official Russell
  3000 total-return series).

If you see an authentication error (for example *“PAM authentication failed”*),
double-check your username/password, make sure any required institutional VPN is
connected, and confirm with WRDS support that your account is enabled for
PostgreSQL access. The script will exit early in that situation so you can retry
after fixing the connection.

Re-run the script whenever you add more constituent files or need to refresh
the data window. Pass `--skip-index` if you plan to supply a different index
return series manually.

## Choosing an EC2 instance type for ReplicaTOR

The Russell 3000 workflow is CPU bound: building distance matrices is
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
  --index russel3000 \
  --solution_name quob \
  --cardinality 300 \
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
vCPUs. The wrapper now forces `OMP_NUM_THREADS` (and the legacy
`REPLICATOR_NUM_CORES`) to the requested value before invoking ReplicaTOR so
that OpenMP-enabled builds actually honour the requested parallelism; if you
need a different limit simply adjust `--replicator_cores`. When experimenting with larger
portfolios (for example `--cardinality 300`), consider extending the time limit,
increasing the core count and upgrading the EC2 instance size so that ReplicaTOR
has enough CPU headroom to search the much larger solution space.

Inactive or constant securities are now filtered out by default before the
distance matrix is built, which prevents ReplicaTOR from wasting medoid slots
on assets that never trade during the training window. Use `--keep_inactive` if
you need to mimic the legacy behaviour (for example when running quick data
health checks).

When ReplicaTOR finishes, the wrapper now saves the medoid assignments to
`prafa/dist_matrix/dist_matrix.clusters.txt`. The `K` medoid indices still live
in `dist_matrix.soln.txt` (or, on older builds, `dist_matrix.soln`); the wrapper
accepts both names. The companion clusters file is derived from the distance
matrix after the solve and is therefore free of the all-zero placeholder array
printed by ReplicaTOR itself. If you set `--replicator_cores` to a power of two
greater than one, the generated `.params` file reflects that choice so ReplicaTOR
can spawn additional worker threads (subject to the compiled binary supporting
multi-threading on the target instance).

## Running the Gurobi baseline

The project also exposes a mixed-integer baseline solved with Gurobi so you can
benchmark QUOB against a deterministic optimiser. Install `gurobipy` inside the
virtual environment (for example via the `pip install` script shipped with the
Gurobi distribution) and make sure the licence has been activated on the EC2
instance.

### Activating a Gurobi licence on the instance

*Standalone and token licences.* If your institution provides a node-locked or
floating token licence, run `grbgetkey` from the `gurobi1200/linux64/bin`
directory (adjust the path if you unpacked Gurobi elsewhere). The tool will
prompt for the host details and write `~/gurobi.lic` by default. Add the
following lines to `~/.bashrc` (or the shell profile you use) so that the Gurobi
libraries and licence are discovered automatically when you reconnect via SSH:

```bash
export GUROBI_HOME=~/Index_Tracking/gurobi1200/linux64
export PATH="$GUROBI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$GUROBI_HOME/lib:$LD_LIBRARY_PATH"
# Only required if you stored the licence somewhere other than $HOME
export GRB_LICENSE_FILE=~/gurobi.lic
```

*WLS Compute Server licences.* If you received Web Licence Service (WLS)
credentials (an Access ID, a Secret Key, and a Licence ID), `grbgetkey` is not
used. Instead, define the environment variables expected by the WLS client
before launching Python or `gurobi_cl`:

```bash
export GUROBI_HOME=~/Index_Tracking/gurobi1200/linux64
export PATH="$GUROBI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$GUROBI_HOME/lib:$LD_LIBRARY_PATH"
export GRB_WLSACCESSID=<your-access-id>
export GRB_WLSSECRET=<your-secret-key>
export GRB_LICENSEID=<your-licence-id>
```

Replace the placeholders with the values listed in the Gurobi portal (for
example for a “WLS Compute Server – Temporary Academic” entitlement). The
variables must be present in every shell session where you import `gurobipy`.
Once they are exported you can verify the setup with `gurobi_cl --version` or by
running `python -c "import gurobipy"`; both commands should succeed without
errors.

Launch the binary quadratic formulation with:

```bash
python main.py \
  --index russel3000 \
  --solution_name gurobi \
  --cardinality 300 \
  --start_date 2014-01-02 \
  --end_date 2023-12-31 \
  --gurobi_time_limit 10800 \
  --gurobi_log_dir logs/gurobi
```

Each rebalance window receives a three-hour time budget by default (10 800 s).
Gurobi streams its own progress log to the console and the wrapper prints a
summary of the exit status (optimal, time limit, suboptimal). If the solver hits
the time limit, the best incumbent solution is kept automatically so that the
subsequent continuous reweighting step can still proceed. Adjust
`--gurobi_time_limit` if you need shorter experiments or, conversely, longer
searches. Any `GurobiError` raised during the solve is rethrown with a concise
message so you can troubleshoot licence or model issues quickly. The optional
`--gurobi_log_dir` flag lets you persist the full solver transcript even if your
SSH session ends; each window writes its own `gurobi_<start>_<end>_<timestamp>.log`
file inside the requested directory and the path is echoed at the end of the
window so you can `tail -f` it later.

### Monitoring long optimisation runs

Long Russell 3000 runs can last several hours per window, especially when both
ReplicaTOR and Gurobi are given the full three-hour budget. To avoid restarting
from scratch after a network hiccup:

1. Launch a `tmux` session before starting the optimisation:

   ```bash
   ssh -i ~/.ssh/id_ed25519 ubuntu@<ip>
   tmux new -s portfolio
   ```

2. Run the usual `python main.py ...` command inside that session.
3. Detach with `Ctrl+b` then `d`; reattach later via `tmux attach -t portfolio`.

Even without `tmux`, the log files generated via `--gurobi_log_dir` record every
iteration so you can inspect what happened just before a disconnect. Remember
that each rebalance window launches a brand new Gurobi instance, so the solver
timer resets to zero at the beginning of every window; this is expected and does
not mean the entire pipeline restarted from scratch.

#### Troubleshooting licence errors

If you still see messages such as *“Restricted license – for non-production use
only”* or *“Model too large for size-limited license”* when running the
portfolio optimisation, Python is falling back to the miniature licence bundled
with the `pip`/`conda` wheels. Double-check the following before re-running the
solver:

* Run `gurobi_cl --version` (or `python -c "import gurobipy; print(gurobipy.gurobi.version())"`).
  The command must succeed without printing the “restricted licence” banner. If
  it does, the environment variables or licence file are either missing or not
  visible in your current shell session.
* For WLS licences, ensure that `GRB_WLSACCESSID`, `GRB_WLSSECRET` and
  `GRB_LICENSEID` are exported **before** activating the virtual environment or
  launching Python. For node-locked or token licences, make sure the
  `gurobi.lic` file lives in `$HOME` (or that `GRB_LICENSE_FILE` points to the
  correct path).
* If you rotated credentials in the Gurobi portal, refresh the environment
  variables or download a fresh `gurobi.lic` with `grbgetkey` before the next
  run.

Only once `gurobi_cl --version` reports your full licence (without the
“restricted” banner) should you launch the Russell 3000 optimisation; otherwise
the solver will abort as soon as the model exceeds the size limits of the demo
licence.

## Analysing Russell 3000 portfolios

After ReplicaTOR (or an alternative solver) has produced a portfolio snapshot
under `results/`, use `scripts/analyze_portfolio.py` to recreate the dashboards
previously built in `analyses_resultats.ipynb`. The helper consumes the weights
generated during rebalancing, rebuilds the out-of-sample returns between each
pair of rebalance dates, and exports both the figures and the underlying time
series.

```bash
cd ~/Index_Tracking
source .venv/bin/activate
python scripts/analyze_portfolio.py \
  --index russel3000 \
  --portfolio quob=results/portfolio_russel3000_quob_300.json \
  --start-date 2014-01-02 \
  --end-date 2023-12-31 \
  --output-dir analyses/russel3000
```

You can pass `--portfolio` multiple times (for example `gurobi=...`) to compare
several optimisation methods side by side. Each run writes:

* `analyses/<label>/returns.csv` – daily portfolio and index returns for the
  backtest window.
* `analyses/<label>/tracking_error.csv` – tracking error measured on each
  out-of-sample window.
* `analyses/<label>/absolute_error.csv` – mean absolute tracking error per
  window.
* `analyses/cumulative_returns.png`, `absolute_differences.png` and
  `error_distribution.png` – the key plots used during the S&P 500 analysis.
* `analyses/summary_metrics.csv` – a compact table with the main statistics for
  every portfolio analysed.

## Troubleshooting large diffs on GitHub

If GitHub refuses to display a diff and reports that the generated diff exceeds
its size limit, the commit usually contains heavy artifacts such as the
`gurobi1200/` solver bundle or raw CSV exports under `financial_data/`. Check
[`docs/troubleshooting_diff_limit.md`](docs/troubleshooting_diff_limit.md) for
practical ways to keep those assets out of your commits or to move them to Git
LFS before opening a pull request.

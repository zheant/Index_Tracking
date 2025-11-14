# Troubleshooting "diff exceeds size limit"

When GitHub tries to display a pull request or commit, it needs to load the diff for every
tracked file that changed. If the diff is larger than GitHub's hard limits (for example, when
a commit introduces hundreds of megabytes of binary data or very large CSV files), the site
will skip rendering the change and you will see the message:

> The generated diff exceeds our size limit and could not be extracted. This can happen when large binaries, minified assets, or an exceptionally extensive refactor are included in a single commit.

In this repository, the most common culprits are heavy assets that live alongside the source
code, such as the `gurobi1200/` directory (contains compiled Gurobi binaries) and the large
CSV datasets under `financial_data/`. Any commit that reintroduces or rewrites those
artifacts can easily produce a diff that is too large for GitHub to preview.

## How to avoid the error

1. **Do not commit generated artifacts.** Before staging changes, run `git status` and make sure
   directories like `financial_data/`, `prafa/`, `results/`, or solver binaries are not marked as
   modified. If they appear, use `.gitignore` or `git restore --staged` to keep them out of the commit.
2. **Split large updates.** If you must version a large dataset, add it in smaller batches across
   multiple commits instead of one massive change.
3. **Consider Git LFS.** For assets that genuinely need to be tracked (e.g., benchmark datasets),
   configure [Git Large File Storage](https://git-lfs.com/) so that the repository history stays light
   and GitHub can render the diffs.
4. **Prune accidental additions.** If you already committed a large binary, remove it with
   `git rm --cached <path>` and add the path to `.gitignore`, then force-push the corrected commit.

Following these guidelines keeps pull requests reviewable and prevents GitHub from rejecting the diff preview.

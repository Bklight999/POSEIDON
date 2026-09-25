# Biology and pharmacology repository search

`mining/bio_repo_search.py` supports four independently resumable stages: `keywords`, `search`, `crawl`, and `merge`. `all` runs them in sequence. Configure GitHub and model credentials first, then run commands from the repository root.

```bash
python -m mining.bio_repo_search keywords
python -m mining.bio_repo_search search --max-keywords 10 --per-combo 20
python -m mining.bio_repo_search crawl --max-repos 20
python -m mining.bio_repo_search merge
```

Remove the small-run limits after checking outputs. Search combines keywords and star buckets, then reviews README content for relevant scientific repositories. Keep GitHub paging concurrency modest to avoid secondary rate limits; model-review concurrency is controlled separately.

Keyword lists cover disciplines, drug discovery, methods, software, species, and diseases. Manually curated terminology bypasses discipline-root filters because terms such as assay names and model organisms may lack recognized biological stems. Ambiguous terms trade higher recall for additional README-review cost.

Default outputs include:

- `workspace/repositories_bio.csv`: selected repository metadata and discipline labels.
- `workspace/bio_scientific_python_files.jsonl`: scientific source-file records.
- `workspace/download_repos_bio_and_medicine/`: downloaded repositories.
- `workspace/merged_scientific_python_files.jsonl`: merged source records.

`BIO_REPO_CSV`, `BIO_NEW_REPOS_DIR`, and `BIO_JSONL_OUT` can override selected locations. Existing records are used to skip previously collected repositories.

Optional domain filtering uses `mining/domain_filter.py` to produce `workspace/repo_domain_labels.jsonl`. `merge --filter-old` retains `KEEP` and `UNCLEAR` labels; unavailable README evidence need not imply a negative domain decision.

Point generation to the merged records and a repository root resolving every referenced source:

```bash
export STEM_STAGE1_JSONL=/path/to/merged_scientific_python_files.jsonl
export STEM_STAGE1_REPOS=/path/to/repositories
```

When combining multiple clone roots, create a stable aggregation directory and preserve repository identity; avoid collisions between equal repository basenames owned by different organizations.

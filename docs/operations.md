# Production operations

Start with the small local run in the README. Distributed production additionally needs configured SSH access, an accessible source tree and dataset pool, model endpoints, and sufficient worker resources.

`pool_and_launch/build_pool.py` selects fresh repositories from configured pools and builds an aggregation directory of source links. Set its source pools, completed-run inputs, and output locations before execution. Preserve the aggregation links for the lifetime of a run.

`pool_and_launch/launch_production.py` is a deployment template. Review its settings and generated remote commands before use: it can stop existing generation processes, install dependencies, and launch workers. Configure host lists, SSH settings, remote working directory, interpreter, pool size, shard size, and model credentials for your own deployment. The supplied host file contains illustrative placeholders only.

Use disjoint repository offsets and limits so each worker handles a separate shard. Round shard sizes upward to cover the pool, exclude unavailable workers before assigning shards, and use distinct output directories. Resume only against the same source records, grouping cache, and output run.

Monitor candidate JSONL, accepted package indexes, rejection statistics, pending QC, worker logs, and free disk space. Keep package rejection reasons separate from execution failures and unavailable model decisions. Tune worker counts together with model request limits and subprocess memory limits; a larger pool or more workers does not guarantee proportional yield.

All private deployment paths must be supplied locally, for example `/path/to/project`, `/path/to/repositories`, and `/path/to/python`. Keep credentials in environment variables and local `.env` files excluded from version control. Do not copy production logs, downloaded repositories, generated datasets, or host lists into the anonymous source release without a separate review.

# Building task images

Packaged tasks include a `dockerfile` describing the execution environment. Review it and build from the task directory so its relative context paths resolve correctly:

```bash
cd /path/to/packaged-task
docker build -f dockerfile -t scientific-task:local .
```

For an optional private registry, authenticate with your own credentials and tag the image for your own namespace:

```bash
docker tag scientific-task:local registry.example.invalid/namespace/scientific-task:latest
docker push registry.example.invalid/namespace/scientific-task:latest
```

Record the resulting image reference alongside the task if required by your evaluation runner. Check dependency installation, image startup, and reference reproduction before distributing an image. Remove local build caches only after verifying which images are still needed.

`qc/build_and_push_images.py` is a separate legacy JSONL-driven image utility. It expects its own repository/instance metadata and configuration; it is not the historical per-task STEM image builder. Review and configure its input format, data location, registry, namespace, and credentials before using it. The historical `build_and_push_stem.py` referenced in development notes is not included in this release.

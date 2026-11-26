# Building DentUnity from the upstream releases

DentUnity is released as a build kit. No image and no upstream annotation file is
re-hosted. The build instructions, the build script and the manifests are in
[`../dataset/`](../dataset/README.md). In short, from the `dentunity/` directory:

```bash
pip install kaggle        # needs a Kaggle account and API credentials
kaggle datasets download -d lokisilvres/dental-disease-panoramic-detection-dataset -p upstream/pano --unzip
kaggle datasets download -d salmansajid05/oral-diseases -p upstream/oral --unzip
python dataset/build/build_dentunity.py --pano-root upstream/pano --oral-root upstream/oral --out dentunity_v1
cat dentunity_v1/verify.json        # every entry of "matches_expected" must be true
```

Read and follow each source page's terms (see [`../dataset/SOURCES.md`](../dataset/SOURCES.md)
and `licence_table.md`). Evaluate on canonical files of the validation and test splits
only. Re-exports may be used for training only when their lineage is in the training split
(`--with-reexports`).

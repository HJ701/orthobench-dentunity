# Additional evaluation controls

From the repository root, run the following with Python 3 and NumPy. The reference output uses NumPy 2.3.5.

    python3 experiments/evaluation_controls.py orthobench dentunity

To save the result, add the option `--output result.json`. The reference run is in [evaluation_controls.json](evaluation_controls.json).

The fitted controls take source width and height. In the photo task, the outcome is whether a classification image also has a detection-task record. In the panoramic task, it is whether a canonical detection image is excluded from the segmentation task. A hierarchical size-frequency classifier is fitted independently on each training split, with a fixed smoothing weight of 16. Validation and test data are used only for evaluation. AP pools tied scores. The 95% paired percentile intervals resample held-out lineages 2,000 times with seed 701. These outcomes concern annotation availability.

The ranking analysis partitions every unordered pair of entries by prediction-head family. It measures rank reversals under changed reference labels and under changed split units. The reference analysis includes fine-tuning as a fourth family. All calculations use the imported versions of the two benchmark releases.

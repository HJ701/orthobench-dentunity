# Reliable evaluation of dental imaging AI

This repository holds the OrthoBench and DentUnity implementations for the MSc thesis of Hamzah Jasim Alomairi at Mohamed bin Zayed University of Artificial Intelligence, *Reliable Evaluation of Dental Imaging AI with OrthoBench and DentUnity*. Alomairi led the machine-learning work on both projects, with contributions from Yutong Xie and Zangir Iklassov. Ian Reid supervised the thesis and collaborated on OrthoBench.

| Directory | Research and code |
| --- | --- |
| [orthobench](orthobench/) | Expert audit of orthodontic reference labels, patient grouping and model comparison |
| [dentunity](dentunity/) | Joint copy-lineage partitions, classification, detection, segmentation and external transfer |

Each directory has its own licensing notices, run instructions and data-access conditions. Run the programs from inside the corresponding directory. The clinical OrthoBench collection is governed, and the DentUnity build kit expects separately obtained upstream images.

## Additional experiments

The [evaluation controls](experiments/) fit two annotation-availability classifiers from the DentUnity training manifests and quantify how often OrthoBench rank reversals occur within and across prediction-head families. The directory includes the executable analysis, complete numerical output and reproduction command.

## Citation

```bibtex
@mastersthesis{alomairi2026reliable,
  author = {Alomairi, Hamzah Jasim},
  title  = {Reliable Evaluation of Dental Imaging AI with OrthoBench and DentUnity},
  school = {Mohamed bin Zayed University of Artificial Intelligence},
  year   = {2026}
}
```

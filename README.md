# Neonatal neuroimaging scratch codes

Research scripts and notebooks for neonatal MRI, from organizing scanner acquisitions to analyzing brain connectivity and training image classifiers. Each study workflow has its own inputs and software requirements.

## Directory guide

| Directory | What you can find there |
| --- | --- |
| [bids/](bids/) | HeuDiConv rules for organizing DICOM series in a BIDS-style layout with study-specific filenames, plus BIDSkit protocol maps for Shanghai and Xiamen. |
| [clinical/](clinical/) | Clinical DICOM preprocessing, cohort metadata scripts, and ANTsRNet experiments. |
| [dti/](dti/) | Nipype workflows for diffusion tensor imaging (DTI) and T1 registration, plus R scripts for group analysis of diffusion measures. |
| [functional/](functional/) | Resting-state fMRI analysis, including regional time series, connectivity matrices, and network plots. |
| [multi_modal/](multi_modal/) | Workflows combining structural, diffusion, and functional MRI, with notebooks for the study figures. |
| [template/](template/) | Shell scripts for preparing input images and building group templates with ANTs. |
| [models/](models/) | MRI classification and segmentation experiments using PyTorch, TensorFlow/Keras, and MXNet. |
| [plot/](plot/) | R code for displaying statistical maps on a 3D brain surface. |

## Try the BIDS heuristic

From the repository root, run this Python 3 example to see how the HeuDiConv rule names a synthetic BOLD series. It needs only the Python standard library.

```bash
cd bids
python3 -B - <<'PY'
from types import SimpleNamespace
from func_heuristic import infotodict

series = SimpleNamespace(series_id="7", series_description="BOLD")
for (template, _, _), series_ids in infotodict([series]).items():
    if series_ids:
        print(series_ids[0], "->", template.format(subject="demo", item=1))
PY
```

The output is `7 -> sub-demo/func/sub-demo_run-1_bold`. The `BOLD` description selects the functional filename template for series 7. The example supplies the subject label `demo` and run number `1`.

## Adapting a study workflow

The T1/DTI script defines two Nipype workflows. Its active run call starts the diffusion workflow with up to 48 local processes. It uses MRtrix3’s `dwidenoise` for denoising, FSL’s `dtifit` for tensor fitting, and ANTs to register the b0 image to a neonatal template. See the [T1/DTI script](multi_modal/process_T1_DTI.py) and [diffusion workflow](multi_modal/workflows.py).

To adapt the script, use `multi_modal/` as the working directory, set `input_dir` and `output_dir`, and supply the tab-delimited cohort table read by `main()`. The input patterns cover T1w and diffusion images plus matching `.bval` and `.bvec` files. The script assigns two fixed subject IDs to `info.iterables` after reading the table. Set that list for your cohort, adjust `n_procs` for your machine, and point the JHU atlas entries to local files in the [atlas configuration](multi_modal/template.json).

For guidance on editing and checking these scripts, see [AGENTS.md](AGENTS.md).

## References

- Image registration with [VoxelMorph](https://github.com/voxelmorph/voxelmorph) and [ANTs](https://github.com/ANTsX/ANTs).
- Diffusion microstructure modeling with [AMICO](https://github.com/daducci/AMICO) and neonatal brain structure segmentation with [NeBSS](https://github.com/PIRCImagingTools/NeBSS).
- Model interpretation with [SHAP](https://github.com/shap/shap) and architecture diagrams with [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet).
- The Keras book is cited in the [local Grad-CAM example](models/classification/cam.py).
- Ceschin et al. (2018). [A computational framework for the detection of subcortical brain dysmaturation in neonatal MRI using 3D Convolutional Neural Networks](https://pubmed.ncbi.nlm.nih.gov/29793060/).

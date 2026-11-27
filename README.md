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

## Resting-state fMRI and T1 analysis

With a cleaned BOLD series, a brain mask, and atlas labels in the same space, `functional.connectome()` averages each region’s signal and writes a region-by-region correlation matrix. The surrounding R script also reads the original study’s cohort and label tables. Starting with only unprocessed T1w and BOLD scans requires changes before this can run reliably across a cohort. [Connectivity calculation](functional/functional_connectome.R#L11)

The HeuDiConv rule recognizes `BRAVO` T1 and `BOLD` series. The T1 branch extracts the brain, denoises it, and enhances contrast. The BOLD branch drops the first five volumes and produces motion estimates, a mean image, a mask, and smoothed images. Downstream scripts perform nuisance regression and temporal filtering, write cleaned 4D images and ALFF, fALFF, and ReHo maps, and calculate network statistics for group comparisons. [Conversion rule](bids/func_heuristic.py#L24) · [T1 and BOLD workflows](multi_modal/process_T1_DTI_head1_three_modal.py#L150) · [Signal cleanup and maps](multi_modal/preprocess_rsbold.R#L84) · [Network analysis](functional/graph_analysis.R#L36)

### Before running a new cohort

- **Select the intended subjects.** The three-modality driver takes only the first metadata row for T1, skips the first 80 subjects for BOLD, and uses fixed BOLD input and output paths. A one-subject BOLD pilot would select nobody with these defaults. [Subject selection and paths](multi_modal/process_T1_DTI_head1_three_modal.py#L233)
- **Check the registration inputs.** The multimodal route aligns T1 and mean BOLD through a diffusion `b0` reference and expects tissue labels in that space. For a cohort with only T1w and BOLD, register mean BOLD to T1w and map tissue masks and atlas labels into BOLD space. [T1 registration and tissue masks](multi_modal/register_t1.py#L86) · [BOLD registration](multi_modal/preprocess_rsbold.R#L58)
- **Fix motion interpolation.** The R loop uses bad frame numbers as voxel-column indices. In a synthetic check at commit `bab94d1`, a spike in one frame across 12 voxels was corrected in only one voxel. [Interpolation loop](multi_modal/preprocess_rsbold.R#L46)
- **Use broadband residuals for fALFF.** The multimodal script band-pass filters before calculating fALFF, whose denominator uses the full frequency spectrum. The separate ALFF script reads residuals before low-pass filtering. [Multimodal calculation](multi_modal/preprocess_rsbold.R#L98) · [Alternative calculation](functional/alff.R#L9) · [fALFF method](https://pubmed.ncbi.nlm.nih.gov/18501969/)

Verification so far covers the source code and the synthetic motion check. For the first participant run, follow one T1w and BOLD scan pair through to a labeled connectivity matrix. Inspect the BOLD–T1 alignment and tissue-mask overlays, and record usable scan duration after motion handling.

Task activation still needs a runnable model with event timings and contrasts. T1 cortical thickness has a FreeSurfer launcher with a fixed subject list and needs separate validation on the intended cohort. [Task-model notes](functional/run.sh#L274) · [FreeSurfer launcher](clinical/nipype_run_recon.py#L31)

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

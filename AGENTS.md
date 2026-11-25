# Working in this repository

Use the [README directory guide](README.md) to find the relevant neonatal MRI study or workflow. Inputs, dependencies, and launch commands vary by directory.

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->

## Editing a workflow

- Trace imports from the script you are changing. Both `dti/nipype_bids/` and `multi_modal/` have their own `workflows.py`, `mrtrix3.py`, and `template.json`. The T1/DTI scripts import their neighboring helpers and open `template.json` from the working directory.
- Check downstream filenames when editing a conversion heuristic. For example, the T1/DTI workflow looks for a diffusion image ending in `_run-1_dti.nii.gz` and matching `.bval` and `.bvec` files. See the [workflow's input patterns](multi_modal/process_T1_DTI.py).
- Use synthetic subject IDs and series descriptions in examples. Keep scans, identifying cohort tables, and generated model checkpoints in the study workspace.

## Checking changes

Choose a check for the workflow you edited. The [README example](README.md#try-the-bids-heuristic) exercises series matching using only Python. From the repository root, `bash -n template/build_bold.sh` checks shell syntax and `python3 -m json.tool multi_modal/template.json > /dev/null` checks the atlas file's JSON syntax.

Read experiment scripts before running them as tests. For example, `multi_modal/test.py` downloads example data and opens a viewer, while `template/test_template.sh` starts ANTs template construction. When reporting a check, say which script ran and which study inputs it used.

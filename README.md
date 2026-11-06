---
language: en
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags:
- experimental
- model-stitching
- go
- weight-editing
library_name: pytorch
---
# Qwen Go learned suture — reference017

**Authors: Trevor Martin–DataStreams Solutions; GPT-6 Astra (AI coauthor).**

Ordinary board text → Qwen hidden layer 8 → learned sender → frozen Moka Go
expert → learned receiver → Qwen layer 28 → Qwen's own generated answer.
There is no parsed board, ground-truth coordinate, or donor move-output bypass
at inference. The text format is constrained (see `example.txt`); this is not
arbitrary-prose board parsing or full-game strength.

This release contains the **017 reference connector** and the retained native
base edit, plus a portable native-only inference script. It does not include
the 4B base weights or the separate packed typed-decision route. No Hugging Face
AutoModel remote-code registration is claimed; use the provided script.

## Public research and code

- [Research article](https://datastreamssolutions.com/research/go-suture/)
- [Summary blog](https://datastreamssolutions.com/blog/stitching-a-go-expert-into-qwen/)
- [Code and reproducibility](https://github.com/tcmartin/qwen-go-suture)
- [Model artifacts](https://huggingface.co/trevdatastreams/qwen3.5-4b-go-suture)

## Run

Use Python 3.12+ and a CUDA PyTorch environment compatible with the pinned
versions. The original work used an RTX3090 and BF16 native SDPA/reference
recurrence. The simplified public loader has CPU component/state checks;
**full-model GPU replay of this extracted loader has not yet been performed.**
This limitation is separate from the saved GPU evaluations of the original
research implementation. Different batch shapes can change numerical results.

```bash
pip install -r requirements.txt
python run_native.py --verify-only
python run_native.py --prompt-file example.txt
```

To use a previously downloaded exact base, add `--base /path/to/Qwen3.5-4B`.
Otherwise the pinned upstream revision downloads automatically (roughly 9 GB).
The script refuses mismatching edited columns and unqualified recurrent kernel
dispatch. It preserves a 24-token greedy output bound and accepts up to 4 prompt
files per call. It is not a throughput benchmark or a general serving system.

The example is a **training** board for usage illustration, not held-out proof.

## What the results show

On repeatedly inspected confirmation examples, reference017 generated 54/96
correct answers (38/64 factual; legal moves 10/16). A direct erasure of 1,152
positional receiver weights reduced mean answer-token NLL by 6.91%, while
correct generated answers fell to 52/96. Removing just their repeated column
mean reduced NLL by 2.57% with 53/96 correct. Those edits are tradeoffs, not
blanket improvements; 017 remains the reference. The later 027 paired-board
training variant increased occupancy 8→9/16 but reduced capture 11→7/16 and
legal moves 10→3/16, so it is not this package's default.

The experiment demonstrates an implemented learned hidden-state connection,
measured interface improvements, and agent-authored direct tensor changes.
Wrong-expert controls often preserve much of the output performance. These
small reused panels do not establish strategic Go transfer, broad reasoning
improvement, novel theoretical methods, or an ability to read beliefs from weights.

## Files and integrity

- `bridge017.safetensors`: learned sender/gate/receiver and frozen donor tensors;
  no optimizer, pickle, RNG, private paths, or credentials.
- `native_base_patch.safetensors`: original/edited L16 downprojection columns;
  required for the native host used in the study.
- `moka/`: pinned original donor binary/manifest and MIT notices.
- `go_suture.py`: unchanged numerical class implementations extracted into a
  portable module; training/operations imports removed.
- `run_native.py`: public native-only loader and guarded greedy generation.
- `suture_config.json`, `source_manifest.json`, `file_manifest.json`: lineage,
  source-extraction hashes, and public-artifact hashes.
- `cpu_package_checks.json`: CPU-only tensor and component verification.

See `NOTICE.md` and `QWEN_LICENSE` for upstream attribution. Published Qwen
RLCD1M/NF4 repositories remain separate and unchanged.

## Reproduce direct edits without gradients

```bash
python test_release.py
python reproduce_edits.py --check
python reproduce_edits.py --variant remove_mean023 --output bridge-remove-mean.safetensors
```

Both scripts run on CPU; no base-model download, training, or GPU is required.
The check compares bitwise with the exported original experiment patch tensors.
The new bridge file is a tradeoff variant; it does not overwrite reference017.
For exploratory inference, supply it to `run_native.py --bridge PATH`; those
portable-loader outputs have not received separate full-model GPU qualification.
See `selected_results.json` for compact adverse and favorable results.

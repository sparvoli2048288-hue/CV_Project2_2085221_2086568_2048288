# Project 2: Joint Detection of AI-Generated Images and Post-Processing Alterations in Real-World Scenarios

Group members:
- Lorenzo Sambucci, 2086568, sambucci.2086568@studenti.uniroma1.it
- Daniele Valenti, 2085221, valenti.2085221@studenti.uniroma1.it
- Davide Sparvoli, 2048288, sparvoli.2048288@studenti.uniroma1.it

## Goal of the project

Given a single input image, one model produces two predictions at the same time:

1. **Real/AI:** whether the image is real or AI-generated;
2. **Transformation:** which post-processing operation was applied: *original*, *transfer* (internet-transmitted) or *redigital* (re-digitized).

We are using two heads (one for each task) and a shared MaxViT-T backbone. This model is trained with a weighted sum of two cross-entropy losses. On top of this base model we experimented three additions (multi-scale features, an SRM channel, learned uncertainty loss weighting) and measured what each one actually contributes through two ablation studies and a cross-backbone check (with ResNet18).

## Our Hardware
All experiments were executed locally. The results refer to the following hardware configuration:
* **GPU:** NVIDIA RTX 5060 Ti (16 GB VRAM) 
* **CPU:** AMD Ryzen 5600x
* **RAM:** 64 GB DDR4 3600MHz
* **Environment:** Python 3.11, PyTorch with CUDA support

## Repository folder
| File | Description |
|---|---|
| `notebook.ipynb` | The full project notebook (data pipeline, models, training, evaluation, ablations), already executed with all outputs |
| `presentation.pdf` | Slides for the presentation of the project |
| `README.md` | This file |

The notebook follows this structure: **Imports, Globals, Utils, Data, Network, Train, Evaluation**, followed by the run sections (unimodal baselines, multi-task models, comparison), the two ablation studies and the ResNet18 cross-backbone check.

## Dataset and subset construction

We use **RRDataset** (Li et al., ICCV 2025): <https://zenodo.org/records/14963880>. It contains real and AI generated images, each one available in three conditions: original, transfer and redigital.

Instead of processing the entire 20 GB dataset, which would lead to unfeasible training and ablation times, we utilize a script to sample a balanced 12000 image subset. This guarantees efficient training loops while preserving the integrity of the multi-task learning evaluation.

1. **Download & extraction with caching:** The dataset is downloaded to `rr_raw/` only if missing; extraction is guarded by a `.extracted` marker file, so re-running the cell never repeats work. If a `RR_subset/` folder already exists the whole build is skipped.
2. **Grouping by source image:** Every image path is scanned for the transformation (the `/original/`, `/transfer/`, `/redigital/` path component) and the class (`/real/`, `/ai/`). The filename is then normalized by stripping transformation and class, so that the three versions of the same source photo collapse onto one key `(class, clean_name)`.
3. **Complete triplets only:** We keep a source image only if all three transformed versions exist. This is what makes perfect balancing possible.
4. **Sampling:** We randomly sample **2,000 triplets per class** (seed 9999). 
5. **Source level split:** Triplets are shuffled and split 70/15/15 into train/val/test. Because the split unit is the triplet, the three versions of a source image always land in the same split, therefore there is no data leakage.
6. **Physical copy:** Files are copied into `RR_subset/<split>/<transformation>/<class>/`. We preferred a real folder tree over an in-memory index: it can be inspected by hand and it survives kernel restarts.

The result is 8,400 training, 1,800 validation and 1,800 test images, with every (split x transformation x class) cell equal by construction.
Right after the build there is a cell that counts the images on disk for every (split x transformation x class) cell and asserts they are all equal within each split. If any cell differs the assert fails and training should not be performed.

## Data loading and preprocessing

`RRFolder` is a `torch.utils.data.Dataset` that walks `RR_subset/<split>/<transformation>/<class>/` and returns `(image, y_fake, y_transform)`, with labels taken from the folder names (`real=0, ai=1`; `original=0, transfer=1, redigital=2`).

We never resize, because we want to keep the pixel-level frequency traces (generation artifacts, compression signatures) that both tasks depend on. Instead:
- **train:** `RandomCrop(224, pad_if_needed=True, padding_mode="reflect")` + horizontal flip (p = 0.5);
- **val/test:** `CenterCrop(224)`.

The order of the operations inside `__getitem__` is the following: PIL geometric transform, `to_tensor` (pixels in [0,1]), optional SRM residual computed on the same crop/flip as the RGB and before ImageNet normalization, per-image standardization of the residual, ImageNet normalization of the RGB channels, channel concatenation.
The SRM residual is a 5×5 high-pass kernel, applied to the luminance of the crop (`0.299 R + 0.587 G + 0.114 B`) with padding 2, so the residual has the same spatial size as the image, so as the three RGB channels.
`build_loaders(use_srm=...)` applies the seeds, creates a seeded `torch.Generator` for the training shuffle and a `worker_init_fn` for the workers, uses `pin_memory` when CUDA is available, and shuffles only the training split. It is called at the start of every experiment so that every run sees exactly the same data order.

## Architecture

The class `Net` lets us run the unimodal baselines, the base multi-task model and the "with novelties" model.

```
Net(backbone, stage_indices, in_chans, tasks, freeze_backbone, dropout)
```

- **Feature extraction:** Instead of calling the backbone end-to-end we iterate it manually: the input goes through `backbone.stem`, then through each of the four stages in `backbone.blocks`, and after every stage we keep the feature map. Each selected stage is global-average-pooled and the pooled vectors are concatenated. MaxViT-T stage widths are 64/128/256/512, so:
  - `stage_indices=(3,)`: **single-scale**, 512-d features (standard approach, we take the last stage);
  - `stage_indices=(0,1,3)`: **multi-scale**, 64+128+512 = **704-d** features. The idea is that forensic traces are low-level and local, so early-stage features should carry signal and traces that the last stage has already lost. We avoid using stage 2 because it should carry only middle level features.
- **Heads:** One `Dropout(0.2) + Linear` per requested task. Passing `tasks=("fake",)`, `("tr",)` or `("fake","tr")` yields the two unimodal baselines and the multi-task model respectively.
- **4-channel stem surgery (SRM variant):** When `in_chans=4`, the first convolution of the stem is replaced by a 4-channel input: the pretrained weights are copied for the RGB channels, and the fourth channel is initialised with the mean of the three RGB kernels (bias copied too). This keeps the pretrained response statistics roughly intact at step 0 instead of injecting a randomly initialised channel that could dominate early gradients.
- **Freeze logic:** With `freeze_backbone=True` all backbone parameters are frozen except the stem (which must stay trainable for the surgery to be learnable). All results in the notebook use full fine-tuning (`FREEZE_BACKBONE=False`, about 30.9 M trainable parameters).

## Training

**Loss:** `JointLoss` supports two modes on the same interface:
- `mode="fixed"`: `L = w_fake · CE_fake + w_tr · CE_tr`
- `mode="uncertainty"`: one learnable scalar `s_i` per task and

  `L = Σ_i exp(−s_i) · CE_i + 1/2 s_i`

  We track log σ_i = exp(s_i/2) every epoch to watch what the weighting actually does. Note that with this formulation the total loss can legitimately go negative (the `1/2 s` terms are unbounded below), so the loss curves of uncertainty runs are comparable within a run but not across weighting modes.

**Optimizer:** AdamW, weight decay 1e-4. With full fine-tuning we use two parameter groups: the backbone at `LR × 0.1` (1e-4) and everything else (heads and the loss parameters `s_i`) at `LR` (1e-3).

**Loop:** `fit()` runs 10 epochs with batch size 32. Per epoch it records train loss, per-task
train accuracy, validation loss (computed by passing the loss function into `evaluate`), per-task validation accuracy and, for uncertainty runs, the σ values. Each experiment cell re-seeds, rebuilds its loaders and its model, trains, evaluates on the test set and plots loss/accuracy curves.

## Evaluation

`evaluate()` per task we compute: accuracy, confusion matrix, per class precision/recall/F1 and
macro-F1. For the Real/AI task it additionally produces the breakdown:

- **accuracy by transformation:** how well fake detection works on original vs transfer vs redigital images;
- **accuracy by (transformation x true class):** the six-cell grid that reveals which class suffers under each post-processing.

Confusion matrices and training curves are plotted for the main models.

## Developed design

| Section | What it trains | Why |
|---|---|---|
| 1a | Unimodal Real/AI (single-scale, RGB) | Baseline required by the track |
| 1b | Unimodal Transform (single-scale, RGB) | Baseline required by the track |
| 2a | Multi-task **with novelties**: multi-scale + SRM + uncertainty | The full proposal |
| 2b | Multi-task **base**: single-scale, RGB, fixed weights at 1.0 | Isolates the joint-training effect; also rung 1 of Ablation A |
| 3 | — | Comparison: (a) unimodal vs multi-task and (b) base vs novelties |
| Ablation A | single-scale RGB → multi-scale RGB → multi-scale+SRM (weights fixed) | Experiments the novelties, one at a time |
| Ablation B | base architecture with weights 1/1, 0.7/0.3, 0.3/0.7, uncertainty | The loss-weighting study required by the track |
| ResNet18 | 2a and 2b repeated on ResNet18 | Checks the findings are not MaxViT-specific |

## How to run

Requirements: Python 3.11, PyTorch + torchvision, numpy, matplotlib, tqdm, Pillow. 

Open the notebook and run it top to bottom. 

Things worth knowing before pressing "run all":

- the first run downloads the 20 GB archive and needs disk space for extraction plus `RR_subset/`, everything is cached afterwards. It is possible that the download stops or gets corrupted, it is possible also to download the dataset from the zenodo site and insert it in the same folder of the notebook. After that rerun the cell and the subset will be created. To rebuild the subset from scratch, delete `RR_subset/` first;
- The notebook is set to run with `FREEZE_BACKBONE=False` 10 EPOCHS, these values can be changed in globals to try different settings for the run;
- on Windows keep `NUM_WORKERS=0`: On Colab and on Linux/MacOS it can be raised to 2 or 4;
- `CROP_SIZE` must stay at 224 because MaxViT's grid attention is resolution specific.

## Results (test set)

Main comparison:

| Model | Real/AI acc | Transform acc |
|---|---|---|
| 1a — unimodal Real/AI | 0.9128 | — |
| 1b — unimodal Transform | — | 0.9839 |
| 2b — multi-task **base** (single-scale, RGB, weights 1/1) | 0.9011 | 0.9928 |
| 2a — multi-task **with novelties** (multi-scale + SRM + uncertainty) | 0.8922 | 0.9906 |

Ablation A — components (fixed 1/1 weights):

| Configuration | Real/AI acc | Transform acc |
|---|---|---|
| single-scale RGB (= 2b) | 0.9011 | 0.9928 |
| multi-scale RGB | 0.9172 | 0.9872 |
| multi-scale + SRM | 0.8906 | 0.9878 |

Ablation B — loss weighting (base architecture):

| Weighting (fake/tr) | Real/AI acc | Transform acc |
|---|---|---|
| fixed 1.0 / 1.0 | 0.9011 | 0.9928 |
| fixed 0.7 / 0.3 | 0.9072 | 0.9878 |
| fixed 0.3 / 0.7 | 0.9078 | 0.9961 |
| learned (uncertainty) | 0.8933 | 0.9950 |

Optimal Combined Configuration (Best of Both):
Applying multi-scale RGB (best from Ablation A) with a fixed 0.3 / 0.7 weighting (best from Ablation B) yields 0.9111 Real/AI accuracy and 0.9889 Transform accuracy.

Cross-backbone check (ResNet18, same pipeline): base 0.8756 / 0.9772, with novelties 0.8667 / 0.9783 — same direction as MaxViT.

Real/AI accuracy by transformation, base model: original 0.9483, transfer 0.8883,
redigital 0.8667; per-cell (real / ai): original 0.933 / 0.963, transfer 0.870 / 0.907,
redigital 0.860 / 0.873.

## What we learned
- **The Optimal Synergy**: By combining our best architectural findings in the two ablations, multi-scale without SRM from ablation A and fixed 0.3/0.7 weighting from ablation B, we achieved a robust configuration, but below each component individual best.
- **Joint training isn't symmetric:** Transform gains from the shared backbone (0.9839 to 0.9928), Real/AI pays for it (0.9128 to 0.9011).
- **Multi-scale helps, SRM hurts:** Bundled together the additions lower both tasks, but Ablation A shows that multi-scale features alone push Real/AI up by 1.6 points (0.9011 to 0.9172, our best Real/AI result, even above the unimodal baseline), and it's the SRM channel that drags it back down (−2.7 points).
- **Equal weights aren't optimal:** 0.3/0.7 beats 1.0/1.0 on both tasks. Uncertainty weighting is the worst option for Real/AI, σ for the transform task collapses during training (0.94 to 0.40), and the easier task ends up dominating the gradient.
- **Originals are the easy case:** In every configuration they're the easiest condition for fake detection; transfer and redigital cost roughly 4–9 points. Where the damage lands depends on the setup: SRM concentrates it on real transformed images, plain RGB spreads it more evenly.
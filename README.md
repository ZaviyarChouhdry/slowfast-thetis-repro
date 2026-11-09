# Reproducing SlowFast 4x16 R50 on THETIS RGB

Assignment 02 (Deep Learning), Bahria University Islamabad.
Muhammad Zaviyar Choudhry (01-134232-254) and Muhammad Ali (01-134232-226).

This repository reproduces the best baseline of:

> E. Hovad, T. Hougaard-Jensen and L. K. H. Clemmensen, "Classification of Tennis Actions Using Deep Learning," arXiv:2402.02545, 2024. DOI: [10.48550/arXiv.2402.02545](https://doi.org/10.48550/arXiv.2402.02545)

The paper trains SlowFast (ResNet-50 backbone) from scratch on THETIS RGB and reports **73.96 % test accuracy** for the 4x16 variant. We re-implement that pipeline in PyTorch + PyTorchVideo and fill the settings the paper leaves out with the defaults of PySlowFast, the framework the authors say they used.

## Repository layout

```
configs/slowfast_4x16_r50_thetis.yaml   every setting, each tagged [paper], [pyslowfast] or [ours]
src/prepare_thetis.py   index videos, cache JPEG frames, 70/20/10 split, dataset figures
src/dataset.py          PySlowFast-style clip sampling, augmentation, slow/fast packing
src/model.py            SlowFast 4x16 R50 (PyTorchVideo) + shape/param/FLOP summary
src/train.py            training loop (SGD, warmup + cosine, AMP, resume, curves)
src/evaluate.py         2 clips x 3 crops test, metrics, confusion matrix, comparison table
notebooks/colab_reproduce_slowfast_thetis.ipynb   Google Colab runner (T4, Drive checkpoints)
notebooks/reproduce_slowfast_thetis.ipynb         Kaggle runner
results/                tables and figures from our run (copied in after training)
```

## 1. Environment

Targets Python 3.10+, PyTorch 2.x, PyTorchVideo 0.1.5, fvcore, one 16 GB GPU (Kaggle T4 or P100). `ffmpeg`/`ffprobe` must be on `PATH`.

```bash
pip install -r requirements.txt
```

The exact versions and GPU used in our run are written to `results/environment.json`.

## 2. Dataset (not included in this repo)

THETIS RGB: 1,980 AVI clips, 12 stroke classes x 165 clips, 55 players (Gourgari et al., CVPR Workshops 2013, DOI [10.1109/CVPRW.2013.102](https://doi.org/10.1109/CVPRW.2013.102)).

The videos are hosted by the dataset authors on GitHub: [THETIS-dataset/dataset](https://github.com/THETIS-dataset/dataset). Only the `VIDEO_RGB` folder is needed (1,980 AVI files, 640x480 at 18 fps, about 3.7 GB). Fetch just that folder:

```bash
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/THETIS-dataset/dataset.git thetis
cd thetis && git sparse-checkout init --no-cone && git sparse-checkout set 'VIDEO_RGB/' && git checkout main
```

Then set `data.root` to `thetis/VIDEO_RGB`. The Colab notebook does this for you.

The dataset is not redistributed here; please follow the dataset authors' terms.

## 3. Reproduce

```bash
CFG=configs/slowfast_4x16_r50_thetis.yaml
OPTS="data.root=/path/to/THETIS data.cache_dir=./cache train.output_dir=./runs/sf4x16 data.split_csv=./runs/splits.csv"

python src/prepare_thetis.py --config $CFG --opts $OPTS   # ~10 min, ~3 GB of JPEG frames
python src/train.py          --config $CFG --opts $OPTS   # ~4-6 h on a T4/P100, resumes automatically
python src/evaluate.py       --config $CFG --opts $OPTS   # writes comparison_table.md, results.json, figures/
```

Any config value can be overridden with `--opts section.key=value` (e.g. `train.batch_size=8` if you run out of memory; the learning rate rescales itself).

**Google Colab (what we used):** open `notebooks/colab_reproduce_slowfast_thetis.ipynb`, set the runtime to T4 GPU, edit `REPO_URL`, and Run all. Checkpoints go to Google Drive, so after a disconnect, Run all again resumes training. A Kaggle version is in `notebooks/reproduce_slowfast_thetis.ipynb`.

## 4. Training settings

| Setting | Value | Source |
|---|---|---|
| Model | SlowFast 4x16, ResNet-50, alpha = 8, beta = 1/8, lateral 5x1x1 conv | paper + PySlowFast |
| Initialisation | random (no pretraining) | paper |
| Input | 64-frame window, 32 fast / 4 slow frames, 224x224 | paper + PySlowFast |
| Train augmentation | short side jittered in [256, 320], random 224 crop, horizontal flip | PySlowFast |
| Split | 70 / 20 / 10, stratified by class, seed 0 | paper (seed ours) |
| Loss | cross-entropy | PySlowFast |
| Optimiser | SGD, momentum 0.9, Nesterov, weight decay 1e-4 (none on BN) | PySlowFast |
| Learning rate | 0.1 x (batch / 64), 34-epoch linear warmup from 0.01 x (batch / 64), then cosine | PySlowFast |
| Epochs | 196, checkpoint with best validation accuracy kept | paper = PySlowFast |
| Batch size | 16 (PySlowFast used 64 across 8 GPUs) | ours, GPU memory |
| Dropout | 0.5 before the classifier | PySlowFast |
| Precision | mixed (AMP) | ours |
| BatchNorm | precise BN statistics recomputed before testing | PySlowFast |
| Test protocol | 2 temporal clips x 3 spatial crops, softmax summed | paper |

Why PySlowFast defaults: the paper names PySlowFast and trains for exactly 196 epochs, which is `MAX_EPOCH` in PySlowFast's `SLOWFAST_4x16_R50.yaml`. The optimiser, learning rate and batch size are not reported, so we take them from that same file.

## 5. Results

**Status: the full training run has not been executed.** Downloading and hosting the complete THETIS video set and ~4–6 GPU-hours of training were not feasible for this submission. The pipeline was verified end to end on synthetic video (every stage runs, checkpoints resume, all tables and figures are written), but no accuracy on THETIS is claimed. Running the three commands in section 3 fills the table below automatically (`comparison_table.md`).

| Evaluation metric | Original paper | Reproduced model | Difference |
|---|---|---|---|
| Train accuracy (%) | 80.39 | not run | – |
| Validation accuracy (%) | 70.20 | not run | – |
| Test accuracy, 2x3 views (%) | 73.96 | not run | – |
| Errors: serve vs serve (% of errors) | 44.4 | not run | – |
| Errors: slice vs volley (% of errors) | 20.4 | not run | – |
| Errors: smash vs serve (% of errors) | 16.7 | not run | – |

Verified model facts (no dataset needed, `python src/model.py --config configs/slowfast_4x16_r50_thetis.yaml`): 33.6 M parameters, 27.8 GFLOPs per 224x224 view.

## 6. Known deviations from the paper

- Batch 16 on one GPU instead of PySlowFast's 64 on eight; learning rate scaled linearly.
- PyTorchVideo implementation of SlowFast rather than PySlowFast itself (same architecture, verified shapes in `model_summary.json`: 33.6 M parameters, 27.8 GFLOPs per 224x224 view).
- The paper's split seed is unknown, so our 70/20/10 test set is a different 198 clips. With ~16 test clips per class, one clip moves per-class accuracy by about 6 points.
- The paper's "beginner" error group needs per-player skill labels that are not in the RGB release, so it is not computed.

## 7. Next step (Assignment 03)

`model.pretrained_path` in the config loads Kinetics weights into the same model, so the proposed improvement (Kinetics initialisation, serve-focused balancing, pose stream) can be compared on the identical split saved in `splits.csv`.

## Citation

```bibtex
@article{hovad2024tennis,
  title   = {Classification of Tennis Actions Using Deep Learning},
  author  = {Hovad, Emil and Hougaard-Jensen, Therese and Clemmensen, Line Katrine Harder},
  journal = {arXiv preprint arXiv:2402.02545},
  year    = {2024},
  doi     = {10.48550/arXiv.2402.02545}
}
```

Also cite THETIS (Gourgari et al., 2013), SlowFast (Feichtenhofer et al., ICCV 2019), PySlowFast and PyTorchVideo if you build on this code.

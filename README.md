# ISL recognition - the pipeline that gave 99.82% (random 80/10/10 split, 36 classes)

NOTE: the 99.82% is a RANDOM-split number. ~91% of its test images had a near-copy in train
(see check_leakage in the other folder), so label it "random split" in any report.

## Folder layout (create it like this)
    <project>/
        src/            <- all .py files from this zip + requirements.txt in <project>/
        dataset/original_images/<class folders 0..9, A..Z>/   <- your Kaggle images
        models/         <- created automatically

## Run (from <project>, with the Python 3.12 venv active)
    pip install -r requirements.txt
    python src/crop_dataset.py       # MediaPipe hand crop  -> dataset/cropped_images
    python src/split_dataset.py      # random 80/10/10      -> dataset/train|validation|test
    python src/train.py              # ~hours; Ctrl+C then rerun to resume
    python src/evaluate.py           # test accuracy + report
    python src/webcam.py             # live test ('m' = mirror, 'q' = quit)

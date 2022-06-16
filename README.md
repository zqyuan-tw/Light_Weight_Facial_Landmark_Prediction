# Light Weight Facial Landmark Prediction
Train a facial landmark detection model with size < 15 MB.

## Install Required Packages
```shell
pip install -r requirements.txt
```

## (Optional) Data Generation
Increase the training data by applying gamma correction and sobel edge detection.
```shell
python get_hybrid.py <--source_dir SOURCE> <--dest_dir DEST>
```
- `--source_dir` : Directory to original training dataset. Inside the source directory, it must contain the annotation file, `annot.pkl`, and the corresponding `.jpg` images.
- `--dest_dir` : Directory to destination. The new `annot.pkl` and the transformed images will be stored in it.

## Usage
```shell
python main.py [--do_train TRAIN] [--do_predict PREDICT]
```
- `--do_train` : Path to training configuration in `.json` format.
    ```json
    // training configuration template
    {
        "model": "MODEL_TYPE",
        "training": {
            "data": "path/to/training/data/directory/",
            "batch_size": 32,
            "num_epoch": 50,
            "learning_rate": 1e-4,
            "checkpoint": 10    // save the model checkpoint every 10 epoches
        },
        "saved_directory": "path/to/the/saved/directory/",
        "validation": {
            "data": "path/to/validation/data/directory/",
            "epoch": 1    // validate the model every 1 epoches and save the model with best validation loss
        }
    }
    ```
    - There are 4 options for `MODEL_TYPE` :
        - `ConvNet` : [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
        - `ELFace` : [Joint Face Detection and Landmark Localization Based on an Extremely Lightweight Network](https://link.springer.com/chapter/10.1007/978-3-030-87358-5_28)
        - `MobileNet` : [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
        - `Transformer` : [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)

- `--do_predict` : Path to inference configuration in `.json` format.
    ```json
    // inference configuration template
    {
        "model": "MODEL_TYPE",
        "testing": {
            "data": "path/to/testing/data/directory/",
            "batch_size": 64
        },
        "saved_path": "path/to/model/weight/pt",
        "output": "path/to/output/txt"
    }
    ```
### Example 
```shell
python main.py --do_train ./config/train.json        # training
python main.py --do_predict ./config/predict.json    # inference
```
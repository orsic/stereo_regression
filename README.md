# End to end Disparity regression

## Setup

```bash
    cp config.json.example config.json 
```

Edit `config.json` to match dataset paths.

## Usage
### Creating tfrecords

```bash
    python create_tfrecords.py --dataset=kitti
```

### Training the model

Create a .json config file:
```json
// experiment.json
{
  "unary": "resnet",
  "cost_volume": "concat",
  "regression": "resnet",
  "classification": "softargmin",
  "loss": "abs",
  "max_disp": 192,
  "num_res_blocks": 7,
  "stem_features": 7,
  "stem_ksize": 3,
  "stem_strides": [2, 2],
  "unary_features": 32,
  "unary_ksize": 3,
  "projection_features": 32,
  "projection_ksize": 3,
  "sparse": true,
  "train": ["/mnt/sda1/morsic/stereo_tfrecords/kitti_train.tfrecords"],
  "train_valid": ["/mnt/sda1/morsic/stereo_tfrecords/kitti_train_valid.tfrecords"],
  "valid": ["/mnt/sda1/morsic/stereo_tfrecords/kitti_valid.tfrecords"],
  "test": ["/mnt/sda1/morsic/stereo_tfrecords/kitti_test.tfrecords"],
  "directory": "/home/morsic/projects/stereo_regression/logs/resnet"
}
```

Train the model using the config file:
```bash
    python train.py --config=experiment.json --epochs=200
```

To plot model losses:
```bash
    python metrics.py --log=logs/resnet/log.txt
```
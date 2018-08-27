# Zalo AI Challenge

## Train

`python data_processinng.py -t train_metadata_csv -o output_filepath_for_pickle`

`python model.py -d dataset_pickle -m model_yaml_path -w weight_path`

## Test

`python test.py -t test_pickle -m model_yaml -w weight_h5`

## Requirements

`sudo apt install ffmpeg`

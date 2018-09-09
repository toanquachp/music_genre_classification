# Zalo AI Challenge

## Train

`python data_processinng.py -t train_metadata_csv -o output_filepath_for_pickle`

`python model.py -d dataset_pickle -m model_yaml_path -w weight_path`

## Test

`python test.py -t test_pickle -m model_yaml -w weight_h5`

## Requirements

`sudo apt install ffmpeg`

## Result

75% accuracy on training set - 4500 songs
70% accuracy on validation set - 500 songs
69% accuracy on test set - 1000 songs

## Dataset

5000 songs for training and 1000 songs for testing were provided by Zalo inc. through the Zalo AI Ch

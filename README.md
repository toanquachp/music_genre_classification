# Zalo AI Challenge - Shiro&Minh

## Members

- Quach Phuong Toan - 0963819039
- Tran Van Minh - 0901380306

## Result

75% accuracy on training set - 4500 songs
70% accuracy on validation set - 500 songs
69% accuracy on test set - 1000 songs

## Dataset

5000 songs for training and 1000 songs for testing were provided by Zalo inc. through the Zalo AI Ch

## Requirements

The program requires `ffmpeg` to run, please run the following command to install it

`sudo apt install ffmpeg`

## Train

Before training, please process the data first by running the `train_processing.py`
Please put the `train.csv` file path and the data-set path to the command, or copy all the song files into the `data/train` folder

`python train_processing.py -t {train.csv} -d {data/train}`

After processing the dataset, the data will be stored as pickle in file `data/data_processing.pickle`

`python train.py`

After training, the model and its weight can be found in the `model` folder as `model.yaml` and `model_weight.h5`

## Test

Before testing, please process the data first by running the `test_processing.py`
Please put the `test.csv` file path and the data-set path to the command, or copy all the song files into the `data/test` folder

`python test_processing.py -t {test.csv} -d {data/test}`

After processing the dataset, the data will be stored as pickle in file `data/test.pickle` 

`python test.py`

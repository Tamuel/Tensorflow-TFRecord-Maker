# TFRecord Maker
Simple source code for converting images and labels into tfrecord.

You can make TFRecord for image and label dataset for Tensorflow.

This use multi threading for boosting speed to make tfrecord.

## How to use
Call ```def make_tfrecord()``` function.
```
    def make_tfrecord(
        image_path,
        label_path,
        train_data_output_path,
        eval_data_output_path,
        shuffle_data,
        number_of_eval)
```
* **Arguments**
  * **image_path** : Path of images. ex) "C:/data/image"
  * **label_path** : Path of labels. ex) "C:/data/label"
  * **train_data_output_path** : Output path for training tfrecord. ex) "C:/data/training.tfrecord"
  * **eval_data_output_path** : Output path for evaluation tfrecord. ex) "C:/data/eval.tfrecord"
  * **shuffle_data** : Whether shuffle image and label or not.
  * **number_of_eval** : Number of evaluation dataset. [# training set = Whole data - number_of_eval]
  
You should change **image_size** and/or **division** in **"dataset_util.py"**.

```
104    image_size = (2048, 2048)
105    division = 1
```
**division** is variable for automatically divide the each image into #division.\
**image_size** is variable for resize original image into image_size.

---
* email : eastern7star@gmail.com

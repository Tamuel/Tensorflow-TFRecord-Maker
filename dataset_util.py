import tensorflow as tf
from multiprocessing.pool import ThreadPool
from threading import Lock
import numpy as np
from time import time
from PIL import Image
import sys
import os
from base_util import is_image, random_select, Timer
from random import shuffle


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def label_preprocessing(label):
    return label


def image_preprocessing(image):
    return image


def run_thread_blocks(number_of_threads, data_list, thread_fn, writer=None, timer=None, return_result=False):
    pool = ThreadPool(processes=number_of_threads)
    data_block_list = list()
    partition_size = int(len(data_list) / number_of_threads)
    for i in range(number_of_threads):
        if i != (number_of_threads - 1):
            data_block_list.append(data_list[i * partition_size:(i + 1) * partition_size])
        else:
            data_block_list.append(data_list[i * partition_size:])

    threads = list()
    for i in range(number_of_threads):
        threads.append(pool.apply_async(thread_fn, (i, data_block_list[i], writer, timer)))

    if return_result:
        outputs = list()
        for i in range(number_of_threads):
            outputs.append(threads[i].get())
        return outputs
    else:
        for i in range(number_of_threads):
            threads[i].get()


def make_tfrecord(image_path, label_path, train_data_output_path, eval_data_output_path, shuffle_data=True,
                  number_of_eval=200):
    """
    Make tfrecord from images and label with multi threads.
    :param image_path: Path of images
    :param label_path: Path of labels
    :param train_data_output_path: Path of train tfrecord
    :param eval_data_output_path: Path of eval tfrecord
    :param shuffle_data: To shuffle data or not
    :param number_of_eval: Number of evaluation set
    """
    if isinstance(image_path, str):
        image_folder_list = [os.path.join(image_path, l) for l in os.listdir(image_path)
                             if is_image(os.path.join(image_path, l))]
    else:
        raise ValueError('Need to provide path of image set in string or list of images')

    if isinstance(label_path, str):
        label_folder_list = [os.path.join(label_path, l) for l in os.listdir(label_path)
                             if is_image(os.path.join(label_path, l))]
    else:
        raise ValueError('Need to provide path of label set in string or list of labels')
    print('Make tfrecord from %s to %s' % (label_path, train_data_output_path))

    image_and_label_list = [[i, j] for [i, j] in zip(image_folder_list, label_folder_list)]
    if shuffle_data:
        shuffle(image_and_label_list)

    eval_list, train_list = random_select(image_and_label_list, number_of_eval, return_unselected=True)

    train_writer = tf.python_io.TFRecordWriter(train_data_output_path)
    eval_writer = tf.python_io.TFRecordWriter(eval_data_output_path)

    # Make list of image data
    print('Start make image and label pairs')

    mutex = Lock()
    mutex2 = Lock()

    image_size = (2048, 2048)
    division = 1

    def load_image_pairs(thread_id, img_and_label_list, writer=None, timer=None):
        global finished_samples
        global time_mean
        global time_start
        if thread_id == 0:
            time_start = time()
            finished_samples = 0
            time_mean = -1
        _height = int(image_size[0] / division)
        _width = int(image_size[1] / division)
        for idx, (img, lab) in enumerate(img_and_label_list):
            image = np.array(Image.open(img).convert('RGB').resize((image_size[0], image_size[1]), Image.ANTIALIAS))
            image = image_preprocessing(image)
            label = np.array(Image.open(label_path).resize((image_size[0], image_size[1]), Image.ANTIALIAS))
            label = label_preprocessing(label)

            divided_images = list()
            divided_labels = list()
            for d_i in range(division * 2 - 1):
                for d_j in range(division * 2 - 1):
                    selected_label = label[
                        d_i * int(0.5 * _height):d_i * int(0.5 * _height) + _height,
                        d_j * int(0.5 * _width):d_j * int(0.5 * _width) + _width
                    ]

                    if np.sum(np.equal(selected_label, 2).astype(np.int32)) == 0:
                        continue

                    divided_images.append(
                        np.array(
                            image[
                                d_i * int(0.5 * _height):d_i * int(0.5 * _height) + _height,
                                d_j * int(0.5 * _width):d_j * int(0.5 * _width) + _width,
                                :3
                            ],
                            dtype=np.uint8
                        )
                    )

                    divided_labels.append(
                        np.array(selected_label, dtype=np.uint8)
                    )

                    if np.shape(divided_images[-1]) != (image_size[0], image_size[1], 3) or \
                        np.shape(divided_labels[-1]) != (image_size[0], image_size[1]):
                        print(np.shape(divided_images[-1]), np.shape(divided_labels[-1]))
                        raise ValueError('Shape of image or label are inappropriate (%s, %s)' % (img, lab))

            if len(divided_images) != len(divided_labels):
                raise ValueError('Number of divided images and labels are different! (%s, %s)' % (img, lab))

            if len(divided_images) > 0:
                for d_i, d_l in zip(divided_images, divided_labels):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'height': int64_feature(_height),
                                'width': int64_feature(_width),
                                'image/raw': bytes_feature(d_i.tostring()),
                                'label/raw': bytes_feature(d_l.tostring())
                            }
                        )
                    )
                    if writer is not None:
                        mutex.acquire()
                        writer.write(example.SerializeToString())  # Write to tfrecord
                        mutex.release()

            mutex2.acquire()
            if timer is not None:
                timer.tick_timer()
            mutex2.release()

    timer = Timer()

    def print_function():
        sys.stdout.write('\rLoad images : [%d/%d][%.2f%%][%d hour %d minute %d second left]' %
                         (timer.current_progress, timer.whole_number_of_data,
                          timer.current_progress / timer.whole_number_of_data * 100,
                          int(timer.remain_time / 3600), int(timer.remain_time / 60) % 60, timer.remain_time % 60))
        sys.stdout.flush()

    timer.print_fn = print_function

    timer.start(len(train_list))
    run_thread_blocks(
        number_of_threads=16,
        data_list=train_list,
        thread_fn=load_image_pairs,
        writer=train_writer,
        timer=timer
    )

    timer.start(len(eval_list))
    run_thread_blocks(
        number_of_threads=16,
        data_list=eval_list,
        thread_fn=load_image_pairs,
        writer=eval_writer,
        timer=timer
    )

    print('\nCreate tfrecord [%s] finished!' % train_data_output_path)
    print('Create tfrecord [%s] finished!' % eval_data_output_path)
    train_writer.close()
    eval_writer.close()


def main(unused_argv):
    make_tfrecord(image_path='C:/data/image',
                  label_path='C:/data/label',
                  train_data_output_path='C:/data/training.tfrecord',
                  eval_data_output_path='C:/data/eval.tfrecord',
                  shuffle_data=True,
                  number_of_eval=200)


if __name__ == '__main__':
    tf.app.run(main=main)

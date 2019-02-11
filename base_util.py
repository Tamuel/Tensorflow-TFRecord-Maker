import os
import numpy as np
import random
from time import time


def file_list(folder_path, sort=True):
    if not os.path.isdir(folder_path):
        raise ValueError('Need to feed path of folder.')

    _list = os.listdir(folder_path)
    _list = [os.path.join(folder_path, l) for l in _list]
    if sort:
        _list.sort()

    return _list


class Timer:

    def __init__(self, as_progress_notifier=True):
        """
        If set as_progress_notifier = True, then it will be use to check progress of some processes.
        If not it will be use to simple timer.
        :param as_progress_notifier:
        """
        self.whole_number_of_data = 0
        self.current_progress = 0
        self.time_moving_average = 0
        self.elapsed_time = 0
        self.remain_time = 0
        self.tick_start_time = 0
        self.is_progress_notifier = as_progress_notifier
        self.timer_ready = False
        self.print_fn = None

    def start(self, number_of_data=None):
        if self.is_progress_notifier:
            if number_of_data is None:
                raise ValueError('You should feed number_of_data.')
            self.whole_number_of_data = number_of_data
            self.current_progress = 0
            self.timer_ready = True
            self.tick_start_time = time()
        else:
            self.tick_start_time = time()
            self.timer_ready = True

    def tick_timer(self):
        if not self.timer_ready:
            raise AttributeError('Need to initialize timer by init_timer().')
        if not self.is_progress_notifier:
            raise AttributeError('You should set as_progress_notifier to True if you want to use tick_timer().')

        if self.current_progress == 0:
            self.time_moving_average = time() - self.tick_start_time
        else:
            tick = time() - self.tick_start_time
            self.elapsed_time = self.current_progress * self.time_moving_average
            self.time_moving_average = (self.elapsed_time + tick) / (self.current_progress + 1)

        self.current_progress += 1
        self.remain_time = (self.whole_number_of_data - self.current_progress) * self.time_moving_average

        if self.print_fn is not None:
            self.print_fn()

        self.tick_start_time = time()

    def check(self, start_timer=True):
        if self.is_progress_notifier:
            raise AttributeError('You should set as_progress_notifier to False if you want to use check().')
        self.elapsed_time = time() - self.tick_start_time

        if self.print_fn is not None:
            self.print_fn()

        if start_timer:
            self.start(self.whole_number_of_data)


def random_integers(min, max, size, exclude: list=None):
    if max < min:
        raise ValueError('Max must be larger than min')
    if exclude is not None:
        if (max - min) >= size + len(exclude):
            rand_list = list(random.sample(range(min, max + 1), size + len(exclude)))
        else:
            rand_list = range(min, max + 1)
        excluded_rand_list = [r for r in rand_list if r not in exclude]
        choices = np.random.choice(excluded_rand_list, size, replace=True if len(excluded_rand_list) < size else False)
    else:
        choices = list(random.sample(range(min, max + 1), size))

    return choices


def random_select(data_list: list, size, exclude: list=None, return_unselected=False):
    selection = random_integers(0, len(data_list) - 1, size, exclude)
    selected = [data_list[i] for i in selection]
    unselected = list()
    for idx, d in enumerate(data_list):
        if idx not in selection:
            unselected.append(d)

    if return_unselected:
        return selected, unselected
    else:
        return selected


def is_image(file_name):
    file_extension = file_name.split('.')[-1]
    if file_extension in ('jpg', 'JPG', 'jpeg', 'JPEG', 'tif', 'TIF', 'png', 'PNG', 'bmp', 'BMP'):
        return True
    else:
        return False

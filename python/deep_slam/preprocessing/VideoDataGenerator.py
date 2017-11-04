from keras.preprocessing.image import *
from keras import backend as K
import numpy as np
import os

class VideoDataGenerator(object):
    def __init__(self,
                 rescale=None,
                 data_format='default'):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.__dict__.update(locals())
        self.rescale = rescale

        if data_format not in {'channels_last', 'channels_first'}:
            raise Exception('data_format should be channels_last or channels_first.'
                            'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        elif data_format == 'channels_last':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

    def flow_from_directory(self):
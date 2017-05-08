# Calculate the computation times for several matrix sizes on GPU
# Author: Christian F. Baumgartner (c.f.baumgartner@gmail.com)

from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']

import numpy as np
import tensorflow as tf
import time
import socket

def get_times(matrix_sizes):

    elapsed_times = []

    for size in matrix_sizes:

        #print("####### Calculating size %d" % size)

        shape = (size,size)
        data_type = tf.float32

        with tf.device('/gpu:0'):
            r1 = tf.random_uniform(shape=shape, minval=0, maxval=1, dtype=data_type)
            r2 = tf.random_uniform(shape=shape, minval=0, maxval=1, dtype=data_type)
            dot_operation = tf.matmul(r2, r1)


        with tf.Session() as session: #config=tf.ConfigProto(log_device_placement=True)
            start_time = time.time()
            result = session.run(dot_operation)
            elapsed_time = time.time() - start_time
            elapsed_times.append(elapsed_time)

    return elapsed_times


if __name__ == "__main__":

    matrix_sizes = range(500,10000,50)
    times = get_times(matrix_sizes)

    print ('---------- Versions: -------------')
    print('Using GPU%s' % os.environ['SGE_GPU'])
    print('Tensorflow version:')
    print(tf.__version__)
    hostname = socket.gethostname()
    print('Hostname: %s' % hostname )
    code_path = os.path.dirname(os.path.realpath(__file__))
    print('Code path: %s' % code_path)

    print('----------- Summary: -------------')

    for ii, size in enumerate(matrix_sizes):
        print("Size: %dx%d, Time: %f secs" % (size,size,times[ii]))

    print('----------------------------------')
    print("Average time: %f" % np.mean(times))
    print('----------------------------------')

    np.savez('%s/%s.npz' % (code_path, hostname), matrix_sizes, times)

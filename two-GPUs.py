import datetime

import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_devices = device_lib.list_local_devices()
    return [x.name for x in local_devices if x.device_type == 'GPU']


def matpow(M, n):
    if n < 1:
        return M
    else:
        return tf.matmul(M, matpow(M, n - 1))


if __name__ == '__main__':
    log_device_placement = False

    gpu_names = get_available_gpus()
    gpu_num = len(gpu_names)

    print('{0} GPUs are detected.'.format(gpu_num))

    device1 = gpu_names[0]
    device2 = gpu_names[1] if int(gpu_num) == 2 else device1
    print('device1: {0}'.format(device1))
    print('device2: {0}'.format(device2))

    '''
    Multi GPUs Usage
    Results on P40
     * Single GPU computation time: 0:00:51.736424
     * Multi GPU computation time: 0:00:28.230548
    '''
    # GPU:0
    with tf.device(device1):
        a = tf.random_normal([10000, 10000])
        b = tf.random_normal([10000, 10000])
        c = tf.random_normal([10000, 10000])

        gpu1 = [matpow(a, 3), matpow(b, 3), matpow(c, 3)]
        sum1 = tf.add_n(gpu1)

    # GPU:0 or GPU:1
    with tf.device(device2):
        d = tf.random_normal([10000, 10000])
        e = tf.random_normal([10000, 10000])
        f = tf.random_normal([10000, 10000])

        gpu2 = [matpow(d, 3), matpow(e, 3), matpow(f, 3)]
        sum2 = tf.add_n(gpu2)

    with tf.device('/cpu:0'):
        sum = sum1 + sum2

    t1_2 = datetime.datetime.now()

    print('Start a session')

    with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
        for i in range(10):
            sess.run(sum)

    t2_2 = datetime.datetime.now()

    print('{0} GPUs computation time: {1}'.format(gpu_num, str(t2_2-t1_2)))


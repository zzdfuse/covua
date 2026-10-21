import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from roop import core

if __name__ == '__main__':
    core.run()

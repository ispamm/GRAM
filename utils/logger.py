import logging
import math


_LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger('__main__')  # this is the global logger


def add_log_to_file(log_path):
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)


class RunningMeter(object):
    """ running meteor of a scalar value
        (useful for monitoring training loss)
    """
    def __init__(self, name=None, val=None, smooth=0.99):
        self._name = name
        self._sm = smooth
        self._val = val

    def __call__(self, value):
        val = (value if self._val is None
               else value*(1-self._sm) + self._val*self._sm)
        if not math.isnan(val):
            self._val = val

    def __str__(self):
        return f'{self._name}: {self._val:.4f}'

    @property
    def val(self):
        if self._val is None:
            return 0
        return self._val

    @property
    def name(self):
        return self._name


"""Historical measured-state lookup, without importing the legacy gen/ CLI."""

import pickle

from LLM.core import common

measured_pkl = None


def check_measured(input_string):
    global measured_pkl
    if measured_pkl is None:
        assert common.HARDWARE_PLATFORM is not None
        with open(f"measured_{common.HARDWARE_PLATFORM}.pkl", "rb") as stream:
            measured_pkl = pickle.load(stream)
    return input_string in measured_pkl

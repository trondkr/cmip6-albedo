import os, sys
import pandas as pd
import xarray as xr
import datetime
import dateutil
from cartopy.util import add_cyclic_point
import os
import logging

# Local files and utility functions
sys.path.append("./subroutines/")

class CMIP6_MODEL():
    def __init__(self, name):
        self.name = name
        self.ocean_vars=[]
        self.ds_dict={}
        self.member_ids=[]

    def description(self):
        logging.info("----- {} -----".format(self.name))
        for ds in self.ds_dict.keys():
            logging.info("Model dataset: {} ".format(ds))
        logging.info("members: {}".format(self.member_ids))
        logging.info("variables: {}".format(self.ocean_vars))
        logging.info("--------------".format(self.name))



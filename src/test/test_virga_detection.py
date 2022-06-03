import unittest

import xarray as xr
import numpy as np

from virga_sniffer import virga_detection as vd


class TestCheckInputConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.input = xr.Dataset({"ze": [1, 2, 3],
                                 "cbh":[1, 2, 3]},
                                coords={'time': [1, 2, 3]})
        self.cfg = {
            'mask_rain': False,
            'mask_vel': False,
            'mask_clutter': False,
        } # custom defaults

    def test_only_ze_and_cbh_needed(self):
        # check for check_input_config
        self.assertIsNone(vd.check_input_config(self.input, self.cfg))

    def test_mandatory_input_missing(self):
        # check if ze missing
        with self.assertRaises(Exception):
            vd.check_input_config(self.input.drop_vars('ze'), self.cfg)
        # check if cbh missing
        with self.assertRaises(Exception):
            vd.check_input_config(self.input.drop_vars('cbh'), self.cfg)

    def test_flag_surface_rain_req(self):
        input1 = self.input
        input2 = self.input.assign({"flag_surface_rain": ('time', [1, 2, 3])})
        cfg = {**self.cfg, 'mask_rain': True}
        # check for check_input_config
        with self.assertRaises(Exception):
            vd.check_input_config(input1, cfg)
        # also check for virga sniffer
        with self.assertRaises(Exception):
            vd.virga_mask(input1, cfg)
        self.assertIsNone(vd.check_input_config(input2, cfg))

    def test_vel_req(self):
        input1 = self.input
        input2 = self.input.assign({"vel": ('time', [1, 2, 3])})
        cfg = {**self.cfg, 'mask_vel': True}
        # check for check_input_config
        with self.assertRaises(Exception):
            vd.check_input_config(input1, cfg)
        # also check for virga sniffer
        with self.assertRaises(Exception):
            vd.virga_mask(input1, cfg)
        self.assertIsNone(vd.check_input_config(input2, cfg))




import unittest
import xarray as xr
import numpy as np

import virga_sniffer.cloud_base_height as cbh


class TestReplaceLayerNaN(unittest.TestCase):
    def setUp(self) -> None:
        sample_data = np.array([[1, 2],
                                [1, 3],
                                [np.nan, 4],
                                [np.nan, np.nan]])
        sample_time = np.datetime64("2014-01-01T12:00:00.000") \
                      + np.arange(sample_data.shape[0]).astype('timedelta64[s]')
        self.input_cbh = xr.DataArray(sample_data, coords={
            'time': sample_time,
            'layer': np.arange(sample_data.shape[1])})

    def test_merge_full_lcl_layer1(self):
        input_lcl = xr.DataArray([2, 2, 1, 1])
        output_cbh, output_mergeidx = cbh.replace_layer_nan(self.input_cbh, input_lcl)
        self.assertTrue(np.all(output_cbh.values[:, 0] == 1))
        self.assertFalse(output_mergeidx[0])
        self.assertFalse(output_mergeidx[1])
        self.assertTrue(output_mergeidx[2])
        self.assertTrue(output_mergeidx[3])

    def test_merge_full_lcl_layer2(self):
        input_lcl = xr.DataArray([2, 2, 1, 1])
        output_cbh, output_mergeidx = cbh.replace_layer_nan(self.input_cbh, input_lcl, layer=1)
        self.assertEqual(output_cbh.values[3, 1], 1)
        self.assertFalse(output_mergeidx[0])
        self.assertFalse(output_mergeidx[1])
        self.assertFalse(output_mergeidx[2])
        self.assertTrue(output_mergeidx[3])

    def test_merge_lcl_with_nan(self):
        # lcl with nan at cbh with valid value
        input_lcl = xr.DataArray([np.nan, 2, 1, 1])
        output_cbh, output_mergeidx = cbh.replace_layer_nan(self.input_cbh, input_lcl)
        self.assertTrue(np.all(output_cbh.values[:, 0] == 1))
        self.assertFalse(output_mergeidx[0])
        self.assertFalse(output_mergeidx[1])
        self.assertTrue(output_mergeidx[2])
        self.assertTrue(output_mergeidx[3])
        # lcl with nan at cbh with nan value
        input_lcl = xr.DataArray([np.nan, 2, np.nan, 1])
        output_cbh, output_mergeidx = cbh.replace_layer_nan(self.input_cbh, input_lcl)
        self.assertTrue(np.isnan(output_cbh.values[2, 0]))
        self.assertTrue(np.all(output_cbh.values[[0, 1, 3], 0] == 1))
        self.assertFalse(output_mergeidx[0])
        self.assertFalse(output_mergeidx[1])
        self.assertFalse(output_mergeidx[2])
        self.assertTrue(output_mergeidx[3])

    def test_output_type(self):
        input_lcl = xr.DataArray([np.nan, 2, np.nan, 1])
        output_cbh, output_mergeidx = cbh.replace_layer_nan(self.input_cbh, input_lcl)
        self.assertIs(type(output_cbh), xr.DataArray)
        self.assertIs(type(output_mergeidx), np.ndarray)


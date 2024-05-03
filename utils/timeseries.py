import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Union
import unittest

@dataclass(init=True)
class TimeseriesWindow:
    size: int
    step: int = 1
    gap: int = 1
    offset: int = 0


def split_timeseries(
    data:Union[list, np.ndarray, pd.Series],
    window_size:int=None,
    step_size:int=1,
    gap_size:int=1,
    offset:int=0,
    window:int=None):
    """
    Splits a time series into windows of size window_size, with a step size of step_size, and a gap of gap_size.
    
    Attributes
    ----------
    data: list, numpy array, pandas series
        The time series data
    window_size: int
        The size of the window
    step_size: int
        The step size between components of the window
    gap_size: int
        The gap size between windows, according to both their starts
    offset: int
        The offset to start the first window
        
    Returns
    -------
    1. In case data is a numpy array with more than 1 dimension
        Numpy array with (ndim + 1)
    2. In case data is a list
        Numpy array with 2 dimensions
    3. In case data is a pandas series
        Tuple of numpy array with 2 dimensions, one containing the index and the other the values 
    
    Examples
    --------
    >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ... split_timeseries(data, 3, 3, 2, 1)
    ... # returns:
    [[2, 5, 8], [4, 7, 10], [6, 9, 12]]
    """
    if window is None:
        if window_size is None:
            raise ValueError("window_size must be provided")
        window = TimeseriesWindow(window_size, step_size, gap_size, offset)
    if isinstance(data, np.ndarray):
        return _split_np_timeseries(data, window)
    elif isinstance(data, list):
        return _split_np_timeseries(np.array(data), window)
    elif isinstance(data, pd.Series):
        return _split_pandas_timeseries(data, window)
    elif isinstance(data, range):
        return _split_np_timeseries(np.array(list(data)), window)
    else:
        raise ValueError("data must be a list, numpy array or pandas series")

def _split_np_timeseries(data:np.ndarray, window:TimeseriesWindow):
    length = data.shape[0]
    current = window.offset
    end = current + (window.size * window.step) - (window.step - 1)
    windows = []
    while end <= length:
        windows.append(data[current:end:window.step])
        current += window.gap
        end = current + (window.size * window.step) - (window.step - 1)
    return np.array(windows)


def _split_pandas_timeseries(data:pd.Series, window:TimeseriesWindow):
    index = data.index.to_numpy()
    values = data.values
    return _split_np_timeseries(values, window), _split_np_timeseries(index, window)


class TestTimeSeriesSplit(unittest.TestCase):
    def test_unit_sequences(self):
        data_list = range(12)
        result_list = [[i] for i in data_list]
        data_np = np.array(data_list)
        expected_np = np.array(result_list)
        data_series = pd.Series(data_list)
        window = TimeseriesWindow(1)
        result_list = split_timeseries(data_list, window=window)
        self.assertTrue(np.array_equal(result_list, expected_np))
        result_np = split_timeseries(data_np, window=window)
        self.assertTrue(np.array_equal(result_np, expected_np))
        result_series = split_timeseries(data_series, window=window)
        self.assertTrue(np.array_equal(result_series[0], expected_np))
    
    def test_windows_size_3_step_2_gap_2_offset_1(self):
        data_list = range(12)
        result_list = [
            [1, 3, 5],
            [3, 5, 7],
            [5, 7, 9],
            [7, 9, 11]
        ]
        data_np = np.array(data_list)
        expected_np = np.array(result_list)
        data_series = pd.Series(data_list)
        window = TimeseriesWindow(3, 2, 2, 1)
        result_list = split_timeseries(data_list, window=window)
        self.assertTrue(np.array_equal(result_list, expected_np))
        result_np = split_timeseries(data_np, window=window)
        self.assertTrue(np.array_equal(result_np, expected_np))
        result_series = split_timeseries(data_series, window=window)
        self.assertTrue(np.array_equal(result_series[0], expected_np))

    def test_windows_size_2_step_4_gap_3_offset_2(self):
        data_list = range(12)
        result_list = [
            [2, 6],
            [5, 9]
        ]
        data_np = np.array(data_list)
        expected_np = np.array(result_list)
        data_series = pd.Series(data_list)
        window = TimeseriesWindow(2, 4, 3, 2)
        result_list = split_timeseries(data_list, window=window)
        self.assertTrue(np.array_equal(result_list, expected_np))
        result_np = split_timeseries(data_np, window=window)
        self.assertTrue(np.array_equal(result_np, expected_np))
        result_series = split_timeseries(data_series, window=window)
        self.assertTrue(np.array_equal(result_series[0], expected_np))

    def test_multivariate(self):
        data_np = np.array(
            [
                [1,     1],
                [2,     4],
                [3,     9],
                [4,     16],
                [5,     25],
                [6,     36],
            ]
        )
        expected_np = np.array(
            [
                [[1, 1], [2, 4], [3, 9]],
                [[2, 4], [3, 9], [4, 16]],
                [[3, 9], [4, 16], [5, 25]],
                [[4, 16], [5, 25], [6, 36]]
            ]
        )
        window = TimeseriesWindow(3, 1, 1)
        result_np = split_timeseries(data_np, window=window)
        self.assertTrue(np.array_equal(result_np, expected_np))
    
    def test_multivariate_step_2(self):
        data_np = np.array(
            [
                [1,     1],
                [2,     4],
                [3,     9],
                [4,     16],
                [5,     25],
                [6,     36],
            ]
        )
        expected_np = np.array(
            [
                [[1, 1], [3, 9], [5, 25]],
                [[2, 4], [4, 16], [6, 36]],
            ]
        )
        window = TimeseriesWindow(3, 2, 1)
        result_np = split_timeseries(data_np, window=window)
        self.assertTrue(np.array_equal(result_np, expected_np))
    
    def test_pandas_index(self):
        # Easter egg
        idx = pd.date_range('2014-07-08', periods=6)
        data_series = pd.Series(range(6), index=idx)
        window = TimeseriesWindow(3, 2, 1)
        expected_idx = np.array([
            [
                '2014-07-08T00:00:00.000000000',
                '2014-07-10T00:00:00.000000000',
                '2014-07-12T00:00:00.000000000',
            ],
            [
                '2014-07-09T00:00:00.000000000',
                '2014-07-11T00:00:00.000000000',
                '2014-07-13T00:00:00.000000000',
            ],
        ], dtype='datetime64[ns]')
        expected_values = np.array([
            [0, 2, 4],
            [1, 3, 5]
        ])
        result_series = split_timeseries(data_series, window=window)
        self.assertTrue(np.array_equal(result_series[0], expected_values))
        self.assertTrue(np.array_equal(result_series[1], expected_idx))

if __name__ == "__main__":
    unittest.main()
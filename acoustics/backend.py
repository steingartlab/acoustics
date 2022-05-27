import glob
import h5py
import numpy as np
import pandas as pd
import pytz
import sqlite3

import resonance.resonance.limit_memory

EST = pytz.timezone('US/Eastern')
DROPS_PREFIX = '/drops/data/'


def _generate_neware_path(id_: str) -> str:
    """Generates full drops path for reading neware data.
    
    Args:
        id_ (str): Experiment id on drops. Ex: 220018-1-5-486
    """
    return f"/drops/anyware/unit_{id_.split('-')[0]}/{id_}.sqlite3"

def _get_capacity_query():
    """Cause I do be lazy like that"""

    return """
        SELECT 
            cycle, 
            test_vol AS vol,
            MAX(test_capchg) AS capacity, 
            MAX(test_capdchg) AS dcap,
            MAX(test_capdchg) / MAX(test_capchg) AS ce
        FROM 
            test
        GROUP BY 
            cycle
        """

def _parse_waves(waves: pd.DataFrame, waves_column: str) -> np.array:
    """Parses waveforms to write to hd5.
    
    Helper function for get_acoustics().

    Args:
        waves (pd.DataFrame): Raw sqlite data, i.e. a dataframe w. amps as list,
            one for each pulse
        waves_column (str): Name of waves column
    
    Returns:
        waves_arr (np.array): Shape no_pulses x no_points_per_pulse
    """

    waves_list = waves[waves_column].apply(eval)
    no_passes = len(waves_list)
    no_samples_per_pass = len(waves_list[0])
    waves_arr = np.zeros((no_passes, no_samples_per_pass))
    for i in range(no_passes):
        for j in range(no_samples_per_pass):
            waves_arr[i, j] = waves_list[i][j]

    return waves_arr

def get_acoustics(exp_id: str, machine: str = 'brix6', waves_column: str = 'amps', n: int = 1) -> np.array:
    """Saves waves as .hd5 (np.array). Allows for faster reading on subsequent runs.
    
    Args:
        exp_id (str): exp_id, excl. drops path. Should correspond to folder within brix-X
        machine (str): Optional. Machine name, corresponding to Drops schema. Defaults to 'brix6'.
        waves_column (str): Optional. The name of the waves column. Defaults to 'amps'.
        n (int): Optional. Supsample by selecting every n-th pass. Defaults to 1 (every pass).

    Returns:
        waves (np.array): Waveforms, shape no_passes x no_samples_pr_pass

    Note:
        Should only be called _after_ an experiment is completed
    """

    path = f'{DROPS_PREFIX}{machine}/{exp_id}/{exp_id}'
    while True:
        try:
            with h5py.File(f'{path}.h5', 'r') as f:
                waves = np.array(f['waves'][:], dtype=np.float16)
            dts = pd.read_csv(
                f'{path}_datetimes.csv',
                parse_dates=['time'],
                infer_datetime_format=True
            )
            break
        except (FileNotFoundError, KeyError): # KeyError for corrupt files
            print("hd5 file doesn\'t exist, generating . . . \n")
            waves_df = _query_acoustics(
                exp_id=exp_id,
                machine=machine,
                additional_params=f'WHERE time % {n} = 0'
            )
            waves = _parse_waves(
                waves=waves_df,
                waves_column=waves_column
            )
            with h5py.File(f'{path}.h5', 'w') as f:
                f.create_dataset('waves', data=waves)
            dts = waves_df.index.to_series()
            dts.to_csv(f'{path}_datetimes.csv', index=False)

    return dts, waves   

def _query_acoustics(exp_id, machine: str = 'brix6', table_name: str = 'table_burst', additional_params: str = None) -> pd.DataFrame:
    """Queries sqlite db 'files' from drops.

    Note difference between this function and get_acoustics(). This is the lower-level function.
    
    Args:
        exp_id (str): exp_id, excl. Drops path. Should correspond to folder within brix-X
        machine (str): Optional. Machine on which experiment was run. Corresponds to drops schema.
            Defaults to 'brix7'.
        table_name (str): Optional. Database table name. Defaults to 'table_burst'.
        additional_params (str): Optional. To further refine query. Defaults to None.

    Returns:
        (pd.DataFrame): UTC-aware dataframe of query.
    """

    if machine is None:
        machine = 'brix7'

    folder = (f'/drops/data/{machine}/{exp_id}/*.sqlite3')
    path = glob.glob(folder)[0]
    connection = sqlite3.connect(path)
    query = f'SELECT * FROM {table_name} {additional_params}'

    data = pd.read_sql(
        sql=query,
        con=connection,
        parse_dates='time',
        index_col='time'
    )

    data.index = data.index.tz_localize(pytz.utc).tz_convert(EST)

    return data

def query_neware(id_, query: str = None):
    """Queries neware data from drops.
    
    Args:
        id_ (str): Experiment id on drops. Ex: 220018-1-5-486
        query (str): Optional. The query to be passed. Defaults to None (i.e. fetch everything).

    Returns:
        neware_data (pd.DataFrame): Yada yada yada
    """

    if query is None:
        query = 'SELECT * FROM test'

    path = _generate_neware_path(id_=id_)
    connection = sqlite3.connect(path)
    neware_data = pd.read_sql(
        sql=query,
        con=connection
    )

    if 'unix_time' in neware_data:
        neware_data['unix_time'] = pd.to_datetime(neware_data['unix_time'], unit='s', utc=True)

    return neware_data

# if __name__ == '__main__':
#     exp_params = experimental_params.exp_params
#     for cell in exp_params.values():
#         params = cell
#         break
#     dt, _ = test_acoustics(params)
#     print(dt.head())

#     params = {'neware_id': '220018-1-5-482'}
#     test_neware(params)

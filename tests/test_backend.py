import matplotlib.pyplot as plt

from pithy3 import *
import backend
import exp_params as experimental_params

def test_acoustics(params):    
    dt, waves = get_acoustics(
        filename=params['acoustics_db'],
        machine=params['machine'],
        waves_column='data',
        n=params['nth']
    )
    print(waves.shape)
    return dt, waves

def test_neware(params):
    neware_data = query_neware(params['neware_id'])
    plt.plot(neware_data.unix_time, neware_data.test_vol)
    showme()
    clf()
    plt.plot(neware_data.unix_time, neware_data.test_cur)
    showme()
    clf()

if __name__ == '__main__':
    exp_params = experimental_params.exp_params
    for cell in exp_params.values():
        params = cell
        break
    dt, _ = test_acoustics(params)
    print(dt.head())

    params = {'neware_id': '220018-1-5-482'}
    test_neware(params)

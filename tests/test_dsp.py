import matploblib.pyplot as plt
import pandas as pd

from pithy3 import *
import resonance.resonance as resonance
import exp_params as experimental_params


def test(params):
    dt, waves = resonance.backend.get_acoustics(exp_id=params['exp_id'],
                                      machine=params['machine'],
                                      waves_column='data',
                                      n=params['nth'])
    dt -= pd.Timedelta('1 hour')  # Daylight savings...
    tof_shift_ = resonance.dsp.tof_shift(waves=waves)
    plt.plot(dt, tof_shift_, 'k', linewidth=1)
    showme()


if __name__ == '__main__':
    exp_params = experimental_params.exp_params
    for name, params_ in exp_params.items():
        if name == 'cell_60':
            params = params_
            break

    test(params)
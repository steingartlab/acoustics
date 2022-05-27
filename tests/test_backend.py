##################
'''TESTS'''

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
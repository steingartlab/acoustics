import numpy as np
import pandas as pd

from pithy3 import *
import resonance.resonance.limit_memory
import exp_params as experimental_params
import figures
import backend
import dsp

exp_params = experimental_params.exp_params
cell_props = experimental_params.cell_props
layers_meta = experimental_params.layers_meta

F = 96500  # Faraday's constant [C/mol]
c_rate = 1 / 3
lithium_density = 534  # kg/m^3
anode_porosity = 0.05  # [ ]


def get_avg_mod(tofs: pd.DataFrame, cell_props: dict, w: list) -> pd.DataFrame:
    """Gets average modulus of battery.
    
    Args:
        tofs (pd.DataFrame): Time-of-Flights
        cell_props (dict): Extrinsic material properties: mass, height, and length.
        w (list): Module thickness (width), either from load cell or caliper measurements.
            Note that it must have same length as tofs
    Returns:
        (pd.DataFrame): The average modulus at every 
    """

    tofs_squared = [(tof * 1E-6)**2 for tof in tofs]  # 1E-6 to match units
    avg_mod = [
        cell_props['mass'] / (cell_props['height'] * cell_props['length'])
    ] * len(tofs)
    for i in range(len(tofs)):
        avg_mod[i] *= w[i] / tofs_squared[i]

    return pd.DataFrame(avg_mod)


def linear_cell_thickness(len_ToF: int,
                          w_f: float,
                          w_i: float = 2E-3) -> np.array:
    """Gets time-dependent module width by interpolating between intial and final (caliper) measurements.
    
    Matches the length of the ToF dataframe.

    Args:
        len_ToF (int): Length of ToF dataframe.
        w_f (float): Battery final thickness
        w_i (float): Optional. Battery initial thickness. Defaults to 2.0E-3 -> standard thickness of
            LiFun anode-free cells.

    Returns:
        (np.array)
    """
    return np.linspace(w_i, w_f, len_ToF)


def sinusoidal_cell_thickness(w_linear: np.array, tofs: list, n_cycles: int,
                              intracycle_max_thickness: float) -> np.array:
    """Expands on the linear thickness model by generating a sinusoidal thickness model.
    
    Args:
        w_linear (np.array): Linear cell model, see linear_cell_thickness().
        tofs (list): Time-of-Flights, see get_absolute_ToF().
        n_cycles (int): Number of cycles.
        intracycle_max_thickness (float): Anode thickness at ToC.

    Returns:
        (np.array): The sinusoidal thickness model.
    """
    linsp = np.linspace(0, len(tofs), len(tofs))
    sin_components = np.sin(np.pi / len(tofs) * n_cycles * linsp)

    return w_linear + (w_linear[-1] - w_linear[0]) * abs(
        sin_components * intracycle_max_thickness)


def parse_load_cell_data(load_cell_data: pd.DataFrame) -> pd.DataFrame:
    """Slow but gets the job done"""
    loads = {}
    for i, row in load_cell_data.iterrows():
        try:
            loads[i] = -float(row)
        except ValueError:
            continue

    return pd.DataFrame.from_dict(data=loads, orient='index')


def get_avg_wavelength(mod: pd.DataFrame, f: float, m: float, l: float,
                       h: float, w: np.array):
    return mod.values * l * w[0] * h / (f * m)


def construct_lifun_layers(cell_props: dict) -> pd.DataFrame:
    """Constructs battery layers from LiFun specs.
    
    Args:
        cell_props (dict): Material properties, incl. individual layer thicknesses.
    
    Returns:
        battery (pd.DataFrame): A dataframe of the layered structure
            index: layer no
            E: stiffness
            x: thickness
            type: layer type, e.g. Cu, separator etc.
    """
    pass


def get_module_thickness_from_load_cell(p: pd.DataFrame):
    # Need to match pressure-w resolution with ToF. Resampling the former prob best way to do it.
    pass


def get_load_cell_data(filename):
    pass


def normalize_moduli(moduli: list, effective_mod: float) -> list:
    """
    
    Args:
        moduli (list): List of pd.DataFrames

    Returns:
        (list): List of pd.DataFrames
    """

    moduli_normalized = []
    for modulus in moduli:
        modulus_normalized = modulus.values - modulus.iloc[
            0].values + effective_mod * 1E9
        moduli_normalized.append(modulus_normalized)

    return moduli_normalized


def get_effective_mod(layers_meta: dict) -> float:
    """Gets effective modulus by using Backus averaging.
    
    See eq here for notation clarification: https://bit.ly/3vBXNRD
    
    """
    sum_E_eff_n = []
    N = 0
    for layer_type, meta in layers_meta.items():
        try:
            E_eff_n = meta['N'] / meta['E']
        except ZeroDivisionError:
            E_eff_n = 0
        sum_E_eff_n.append(E_eff_n)
        N += meta['N']

    mod_eff = N / sum(sum_E_eff_n)

    return mod_eff


def get_anode_thickness_change_rate(I: float,
                                    rho: int,
                                    A: float,
                                    z: int = 1,
                                    M: float = 6.9E-3) -> float:
    """Calculates the rate of thickness change from Faraday's Law.
    
    Args:
        I (float): Current [A]
        rho (int): The anode density. Note that this value might/should be slightly different from ideal as
            the plating might be nonideal and thus lower.
        A (float): Module area [m^2].
        z (int): Optional. Charge on species. Defaults to 1 (Li)
        M (float): Optional. Molar mass of species, in kg/m^3. Defaults to 6.9E-3 (Li)

    Returns:
        (float)
    """

    return I * M / (F * z * A * rho)


def layered_mod(tofs: pd.DataFrame, cell_props: dict, w: list) -> pd.DataFrame:
    anode_thickness = get_anode_thickness()

    return


def get_modulus(params):
    cell_area = cell_props['height'] * cell_props['length']
    anode_thickness_change_rate = get_anode_thickness_change_rate(
        I=cell_props['capacity'] * c_rate,
        A=cell_area,
        rho=lithium_density * (1 - anode_porosity))
    anode_final_thickness = anode_thickness_change_rate * 3600 / c_rate
    print(anode_final_thickness / 16)
    # electrolytes.append(params['electrolyte'])
    # LOAD CELL
    # load_cell_data = get_load_cell_data()#query_drops(db=LOAD_CELL_DB)
    # load_cell_data = parse_load_cell_data(load_cell_data)

    # ToF
    dt, acoustic_amplitudes = backend.get_acoustics(exp_id=params['exp_id'],
                                                    machine=params['machine'],
                                                    waves_column='data',
                                                    n=params['nth'])
    dt -= dt.iloc[0]
    dt /= pd.Timedelta('1 hour')

    tofs, trigger_indices = dsp.absolute_tof(acoustic_amplitudes, params)
    # print(f'anode ToC thickness: {round(anode_final_thickness*1E6)} um')

    # plot_tofs(tofs)

    # index = 100  # randomly selected
    # figures.plot_trigger(
    #     acoustic_amplitudes=acoustic_amplitudes[index,:],
    #     trigger_index=trigger_indices[index]
    # )

    # if params['mode'] == 'pulse/echo':
    #     tofs = [tof/2 for tof in tofs]  # For pulse/echo

    # w = get_module_thickness_from_load_cell(p=p)
    w_linear = linear_cell_thickness(w_i=params['w'][0],
                                     w_f=params['w'][-1],
                                     len_ToF=len(tofs))
    avg_mod = get_avg_mod(tofs=tofs, cell_props=cell_props, w=w_linear)

    w_sinusoidal = sinusoidal_cell_thickness(
        w_linear=w_linear,
        tofs=tofs,
        n_cycles=params['n_cycles'],
        intracycle_max_thickness=anode_final_thickness * 1E3)
    sinusoidal_mod = get_avg_mod(tofs=tofs,
                                 cell_props=cell_props,
                                 w=w_sinusoidal)

    figures.plot_thickness_models(dt=dt,
                                  models=[w_linear, w_sinusoidal],
                                  labels=['Linear', 'Faraday Sinusoidal'])

    # f = 2.5E6
    # wavelength = get_avg_wavelength(
    #     mod=mod,
    #     f=f,
    #     l=LENGTH,
    #     h=HEIGHT,
    #     m=MASS,
    #     w=w
    # )
    # print(wavelength.shape)
    # clf()
    # plt.plot(wavelength)
    # showme()

    #     break
    # effective_mod = get_effective_mod(layers_meta)
    # moduli = normalize_moduli(moduli, effective_mod)

    return dt, sinusoidal_mod

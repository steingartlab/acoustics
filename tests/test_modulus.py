import modulus

from pithy3 import *
import exp_params as experimental_params
import figures


def main():
    names = []
    electrolytes = []
    dts = []
    moduli = []
    for name, params_ in exp_params.items():
        dt, modulus = modulus.get_modulus(params_)
        names.append(name)
        electrolytes.append(params_['electrolyte'])
        dts.append(dt)
        moduli.append(modulus)

    effective_mod = modulus.get_effective_mod(layers_meta)
    moduli = modulus.normalize_moduli(moduli, effective_mod)
    figures.plot_modulus(dts, moduli, names, electrolytes)

    showme()

if __name__ == '__main__':
    main()
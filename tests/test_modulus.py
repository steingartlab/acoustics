import modulus

from pithy3 import *
import resonance.resonance as resonance
import exp_params as experimental_params


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

    effective_mod = resonance.modulus.get_effective_mod(layers_meta)
    moduli = resonance.modulus.normalize_moduli(moduli, effective_mod)
    resonance.figures.plot_modulus(dts, moduli, names, electrolytes)

    showme()

if __name__ == '__main__':
    main()
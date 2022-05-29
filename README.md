# acoustics

This is the core codebase needed for analyzing acoustic experiments. The software for conducting experiments is not publically available, more out of lack of refactoring it in an presentable manner than anything else.

It has the functionality needed to load, parse and perform standard acoustic analysis on data from both potentiostats and oscilloscopes. Analysis techniques include &mdash; but are not limited to &mdash; amplitude analysis, ToF-shifts, absolute ToF, Young's Modulus, and capacity.

It also includes a hodgepodge of classes and functions to construct and assemble figures.

It is designed to be called from the omnipotent [pithy](https://github.com/dansteingart/pithy) and use the file management system [drops](https://github.com/dansteingart/drops), both written by proftron Dan himself. 

```
./
├── acoustics
│   ├── backend.py
│   ├── dsp.py
│   ├── figures.py
│   └── modulus.py
├── LICENSE
├── README.md
└── tests
    ├── test_backend.py
    ├── test_dsp.py
    └── test_modulus.py
```
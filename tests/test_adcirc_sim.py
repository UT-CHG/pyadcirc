#!/work/06307/clos21/ls6/mambaforge/envs/pya-test/bin/python

from pyadcirc.sim.ADCIRCSim import BaseADCIRCSimulation

if __name__ == "__main__":
    import sys
    import logging

    simulation = BaseADCIRCSimulation(
        name="ADCIRCSim",
        log_config={
            "output": "ADCIRCSim-log",
            "fmt": "{message}",
            "level": logging.DEBUG,
        },
    )
    simulation.run(
        args={
            "input_dir": "/work/06307/clos21/pub/adcirc/inputs/ShinnecockInlet/mesh/def",
            "exec_dir": "/work/06307/clos21/pub/adcirc/execs/ls6/v56_beta/",
            "cp": 10,
            "wp": 0,
        }
    )

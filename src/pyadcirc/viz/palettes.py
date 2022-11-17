# FigureGenPalettes

Default = {
    "name": "Default",
    "desc": "Default sequential colormap to use for most nodal and output value plots.",
    "num_pals": 1,
    "pals": {
        "DefaultFull": {
            "range": [0.0, 1.0],
            "intervals": [
                [0.0, 0.0, 0.0, 139.0],
                [0.0942, 0.0, 0.0, 255.0],
                [0.1945, 125.0, 158.0, 192.0],
                [0.31, 98.0, 221.0, 221.0],
                [0.4164, 0.0, 210.0, 0.0],
                [0.5137, 255.0, 255.0, 0.0],
                [0.6079, 255.0, 215.0, 0.0],
                [0.7052, 255.0, 104.0, 32.0],
                [0.7872, 251.0, 57.0, 30.0],
                [0.8541, 232.0, 0.0, 0.0],
                [0.9119, 179.0, 0.0, 0.0],
                [0.96, 221.0, 0.0, 221.0],
            ],
        }
    },
}

TopoBlueGreen = {
    "name": "TopoBlueGreen",
    "desc": "Diverging colormap best used for topography/bathymetry plots",
    "num_pals": 1,
    "pals": {
        "TopoBlueGreenFull": {
            "range": [0.0, 1.0],
            "intervals": [
                [0.0, 0, 0, 130],
                [0.25, 0, 0, 255],
                [0.5, 0, 255, 255],
                [0.62, 107, 195, 135],
                [0.81, 0, 128, 0],
                [1.0, 0, 64, 0],
            ],
        }
    },
    "intervals": [
        (-100.0, 0),
        (-50.0, 0),
        (-20.0, 0),
        (-10.0, 0),
        (-5.0, 0),
        (-2.0, 0),
        (-1.0, 0),
        (-0.5, 0),
        (-0.2, 0),
        (-0.1, 1),
        (0.1, 0),
        (0.2, 0),
        (0.5, 0),
        (1.0, 0),
        (2.0, 0),
        (5.0, 0),
        (10.0, 0),
        (20.0, 0),
    ],
}

TopoBlueGreenPurple = {
    "name": "TopoBlueGreenPurple",
    "desc": "Diverging cmap best for topo/bathymetry plots with deep ranges",
    "num_pals": 1,
    "pals": {
        "TopoBlueGreenFull": {
            "range": [0.0, 1.0],
            "intervals": [
                [0.0, 238, 47, 157],
                [0.06666666666666667, 115, 24, 143],
                [0.13333333333333333, 57, 13, 136],
                [0.2, 18, 6, 134],
                [0.26666666666666666, 16, 0, 255],
                [0.3333333333333333, 56, 84, 254],
                [0.4, 81, 166, 255],
                [0.4666666666666667, 112, 251, 254],
                [0.5333333333333333, 120, 255, 240],
                [0.6, 226, 226, 226],
                [0.6666666666666666, 100, 194, 108],
                [0.7333333333333333, 73, 169, 61],
                [0.8, 55, 145, 31],
                [0.8666666666666667, 45, 123, 24],
                [0.9333333333333333, 35, 98, 19],
                [1.0, 26, 76, 13]
            ],
        }
    },
    "intervals": [
        (-7500.0, 0),
        (-5000.0, 0),
        (-2000.0, 0),
        (-750.0, 0),
        (-500.0, 0),
        (-250.0, 0),
        (-100.0, 0),
        (-25.0, 0),
        (-1.0, 0),
        (0.0, 1),
        (0.1, 0),
        (1.0, 0),
        (2.0, 0),
        (5.0, 0),
        (10.0, 0),
        (25.0, 0),
        (50.0, 0)
    ],
}

TopoRainbow = {
    "name": "TopoRainbow",
    "desc": "Rainbow palette for bathymetry/topography plots.",
    "num_pals": 1,
    "pals": {
        "TopoRainbowFull": {
            "range": [0.0, 1.0],
            "intervals": [
                [0.0, 0, 0, 139],
                [0.3, 0, 0, 255],
                [0.5, 125, 158, 192],
                [0.55, 98, 221, 221],
                [0.6071, 0, 210, 0],
                [0.6327, 255, 255, 0],
                [0.6939, 255, 215, 0],
                [0.7551, 255, 104, 32],
                [0.8163, 251, 57, 30],
                [0.8776, 232, 0, 0],
                [0.9388, 179, 0, 0],
                [1.0, 221, 0, 221],
            ],
        }
    },
    "intervals": [
        (-100.0, 0),
        (-60.0, 0),
        (-30.0, 0),
        (-20.0, 0),
        (-10.0, 0),
        (-5.0, 0),
        (-2.0, 0),
        (-1.0, 0),
        (0.0, 0),
        (1.0, 0),
        (2.0, 0),
        (5.0, 0),
        (10.0, 0),
        (20.0, 0),
        (30.0, 0),
    ],
}

GridSize = {
    "name": "GridSize",
    "desc": "Sequential palette for visualizing mesh element size differences",
    "num_pals": 1,
    "pals": {
        "GridSizeFull": {
            "range": [0.0, 1.0],
            "intervals": [
                [0.04, 221, 0, 221],
                [0.0881, 179, 0, 0],
                [0.1459, 232, 0, 0],
                [0.2128, 251, 57, 30],
                [0.2948, 255, 104, 32],
                [0.3921, 255, 215, 0],
                [0.4863, 255, 255, 0],
                [0.5836, 0, 210, 0],
                [0.69, 98, 221, 221],
                [0.8055, 125, 158, 192],
                [0.9058, 0, 0, 255],
                [1.0, 0, 0, 139],
            ],
        }
    },
    "intervals": [
        (20.0, 0),
        (40.0, 0),
        (60.0, 0),
        (80.0, 0),
        (100.0, 0),
        (200.0, 0),
        (400.0, 0),
        (600.0, 0),
        (800.0, 0),
        (1000.0, 0),
    ],
}

Decomp = {
    "name": "GridDecomposition",
    "desc": "Categorical colormap for visualizing grid domain decompoisitons",
    "num_pals": 1,
    "pals": {
        "DecompFull": {
            "range": [0.0, 1.0],
            "intervals": [
                [0.0, 0.0, 0.0, 255.0],
                [0.2, 0.0, 255.0, 0.0],
                [0.4, 255.0, 255.0, 0.0],
                [0.6, 255.0, 127.0, 0.0],
                [0.8, 255.0, 0.0, 0.0],
                [1.0, 255.0, 0.0, 255.0],
            ],
        }
    },
}
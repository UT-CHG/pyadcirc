"""
viz - utils.py

Vizualization utility functions

"""
from pathlib import Path
from typing import AnyStr, Callable, List, Tuple, Union
import imageio as iio
import numpy as np
from pygifsicle import optimize


def generate_gif(
    name: str,
    gen_image: Callable,
    args: List,
    repeat: Union[List, int] = None,
    figsize: Tuple = (12, 6),
    build_dir: str = None,
    hold_end: float = 0.25,
):
    """
    Generate GIF

    Given a callable and a list of arguments to pass to the callable, build a
    gif using sequence of images produced by `gen_image` callabe function.

    Parameters
    ----------

    Returns
    -------


    """

    build_dir = Path.cwd() if build_dir is None else Path(build_dir)

    gif_images_path = build_dir / ".gif_images"
    gif_images_path.mkdir(exist_ok=True)
    gif_path = build_dir / name

    images = []
    gif_images = []

    if repeat is None:
        repeat = np.array([1]).repeat(len(args))
    elif type(repeat) == int:
        repeat = np.array([repeat]).repeat(len(args))

    if args is None:
        args = range(len(data["time"]))
    for i, t in enumerate(args):
        filename = str(gif_images_path / f"{t}")
        filename = gen_image(t, filename)
        images.append(filename)
        for j in range(repeat[i]):
            gif_images.append(filename)

    if hold_end > 0:
        num_extra = int(hold_end * len(gif_images))
        for i in range(num_extra):
            gif_images.append(filename)

    with iio.get_writer(str(gif_path), mode="I") as writer:
        for i in gif_images:
            writer.append_data(iio.imread(i))

    optimize(gif_path)

    for i in images:
        Path(i).unlink()

    return gif_path

import time
import asciichartpy
import numpy as np
import sys

import pdb


def text_line_plot(
    index,
    vals,
    threshold=1.0,
    width=80,
    title="Test",
    clear=True,
    hold_end=True,
    scale_to_fit=False,
    **kwargs,
):
    """
    Generate text for line plot to print
    """
    if scale_to_fit:
        step = int(len(vals) / width)
        step = 1 if step == 0 else step
        selection_idxs = list(range(0, len(vals), step))
        to_plot_vals = vals[selection_idxs]
        index = index[selection_idxs]
    else:
        to_plot_vals = vals

    num_points = len(to_plot_vals)
    width = num_points if num_points < width else width
    hw = int(width / 2 - 1)
    zeroes = list(np.zeros(width))
    trigger = list(threshold * np.ones(width))
    cfg = {
        "format": "{: 3.3f}",
        "height": 10,
        "colors": [asciichartpy.yellow, asciichartpy.red, asciichartpy.green],
    }
    cfg.update(kwargs)
    plot_indices = (
        [(num_points - hw)] if scale_to_fit else range(hw, num_points - hw, 1)
    )
    delete_lines = 13 if not hold_end else 16
    for i in plot_indices:
        if len(plot_indices) > 1:
            for j in range(delete_lines):
                print(str.ljust("\033[A", 100, " ") + "\033[A")
        # Overlay the two lines
        end = (i + hw - 1) if (i + hw - 1) < num_points else num_points - 1
        print(str.ljust(f"{title}", 100, " "))
        print(str.ljust(f"Time:{index[i-hw]} - {index[end]}", 100, " "))
        to_plot = list(to_plot_vals[i - hw : end])
        chart = asciichartpy.plot([zeroes, trigger, to_plot], cfg=cfg) + "\n"
        sys.stdout.buffer.write(chart.encode("utf-8"))
        sys.stdout.flush()
        time.sleep(0.05)

    if hold_end:
        _ = input("\n============ Press <Enter> to continue ============\n")

    if clear:
        for j in range(delete_lines):
            print(str.ljust("\033[A", 100, " ") + "\033[A")

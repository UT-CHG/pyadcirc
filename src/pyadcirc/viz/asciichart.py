import time
import asciichartpy
import numpy as np
import sys

import pdb


def text_line_plot(
    index,
    vals,
    threshold=1.0,
    height=10,
    width=80,
    fmt="{: 3.3f}",
    title="Test",
    clear=True,
    hold_end=True,
    scale_to_fit=False,
    prompt=None,
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
    thresholds = [threshold] if not isinstance(threshold, list) else threshold

    colors = [asciichartpy.green]
    to_plot_lines = (len(thresholds) + 2) * [None]
    to_plot_lines[0] = list(np.zeros(width))  # origin zero line
    for i, t in enumerate(thresholds):
        to_plot_lines[i + 1] = list(t * np.ones(width))
        colors.append(asciichartpy.lightyellow)
    if len(thresholds) >= 2:
        colors[-2] = asciichartpy.yellow
    colors[-1] = asciichartpy.red
    colors.append(asciichartpy.lightblue)
    cfg = {
        "format": fmt,
        "height": height,
        "colors": colors,
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
        to_plot_lines[-1] = list(to_plot_vals[i - hw : end])
        chart = asciichartpy.plot(to_plot_lines, cfg=cfg) + "\n"
        sys.stdout.buffer.write(chart.encode("utf-8"))
        sys.stdout.flush()
        time.sleep(0.05)

    res = 'N'
    if hold_end:
        if prompt is None:
            _ = input("\n============ Press <Enter> to continue ============\n")
        else:
            res = input(prompt)

    if clear:
        for j in range(delete_lines):
            print(str.ljust("\033[A", 100, " ") + "\033[A")

    return res

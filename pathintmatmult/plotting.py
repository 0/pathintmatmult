"""
Convenience functions for plotting the generated data.
"""

import matplotlib.pyplot as plt


def plot2d(data: '[[X]]', x_range, y_range, out_path, *, x_label=None, y_label=None, colormap='jet', colorbar=True):
    """
    Plot the data as a heat map.

    The resulting image is saved to out_path.

    Parameters:
      data: Two-dimensional array of numbers to plot.
      x_range: Tuple containing the min and max values for the x axis.
      y_range: Tuple containing the min and max values for the y axis.
      out_path: The path to the file where the image should be written. The
                extension determines the image format (e.g. pdf, png).
      x_label: Label for the x axis.
      y_label: Label for the y axis.
      colormap: matplotlib colormap to use for the image.
      colorbar: Whether to display the colorbar.
    """

    fig = plt.figure()
    ax = fig.gca()

    img = ax.imshow(data, cmap=colormap, origin='lower', extent=(x_range + y_range))

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    if colorbar:
        fig.colorbar(img, drawedges=False)

    fig.savefig(out_path, bbox_inches='tight', transparent=True)

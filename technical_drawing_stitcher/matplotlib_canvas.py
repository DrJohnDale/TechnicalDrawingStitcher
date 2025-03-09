from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes_initial = self.fig.add_subplot(221)
        self.axes_to_merge = self.fig.add_subplot(222)
        self.axes_merged = self.fig.add_subplot(212)
        super().__init__(self.fig)


def create_matplotlib_layout(main_window):
    canvas = MplCanvas(main_window, width=5, height=4, dpi=100)
    toolbar = NavigationToolbar(canvas, main_window)
    layout_image = QtWidgets.QVBoxLayout()
    layout_image.addWidget(toolbar)
    layout_image.addWidget(canvas)
    return layout_image, canvas

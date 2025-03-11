import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes_initial = self.fig.add_subplot(221)
        self.axes_to_merge = self.fig.add_subplot(222)
        self.axes_merged = self.fig.add_subplot(212)
        self.set_rectangle_selector_initial()
        self.set_rectangle_selector_to_merge()
        super().__init__(self.fig)


    def set_rectangle_selector_initial(self):
        self.rs_initial = RectangleSelector(self.axes_initial, line_select_callback,
                                            button=plt.MouseButton.RIGHT,  # don't use middle button
                                            interactive=True, useblit=True, spancoords="pixels")


    def set_rectangle_selector_to_merge(self):
        self.rs_to_merge = RectangleSelector(self.axes_to_merge, line_select_callback,
                                            button=plt.MouseButton.RIGHT,  # don't use middle button
                                            interactive=True, useblit=True, spancoords="pixels")

def create_matplotlib_layout(main_window):
    canvas = MplCanvas(main_window, width=5, height=4, dpi=100)
    toolbar = NavigationToolbar(canvas, main_window)
    layout_image = QtWidgets.QVBoxLayout()
    layout_image.addWidget(toolbar)
    layout_image.addWidget(canvas)
    return layout_image, canvas


def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print(" The button you used were: %s %s" % (eclick.button, erelease.button))
import matplotlib
matplotlib.use('Qt5Agg')
import sys
from PyQt5 import QtWidgets
from technical_drawing_stitcher.matplotlib_canvas import create_matplotlib_layout
from technical_drawing_stitcher.buttons_and_labels import get_buttons_and_labels_layout
from technical_drawing_stitcher.core import Core


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core = Core(self)
        mpl_layout, self.core.canvas = create_matplotlib_layout(self)
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(mpl_layout)
        layout.addLayout(get_buttons_and_labels_layout(self.core))

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.showMaximized()
        self.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()

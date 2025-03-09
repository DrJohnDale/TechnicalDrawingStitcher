from technical_drawing_stitcher.matplotlib_canvas import MplCanvas
import cv2
import numpy.typing as npt
import typing
from matplotlib.patches import ConnectionPatch
from PyQt5 import QtWidgets
from technical_drawing_stitcher.padded_warp import warpAffinePadded
import numpy as np


class Core:
    def __init__(self, main_window: QtWidgets.QMainWindow):
        self.canvas: typing.Union[MplCanvas, None] = None
        self.im_initial: typing.Union[cv2.Mat, None] = None
        self.im_to_merge: typing.Union[cv2.Mat, None] = None
        self.im_merged: typing.Union[cv2.Mat, None] = None
        self.affine: typing.Union[cv2.Mat, None] = None
        self.matched1: typing.Union[npt.NDArray, None] = None
        self.matched2: typing.Union[npt.NDArray, None] = None
        self.main_window: QtWidgets.QMainWindow = main_window

    def update_plot(self, axes, im):
        axes.cla()
        if im is not None:
            axes.imshow(im)
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def update_initial_plot(self):
        self.update_plot(self.canvas.axes_initial, self.im_initial)

    def update_to_merge_plot(self):
        self.update_plot(self.canvas.axes_to_merge, self.im_to_merge)

    def update_merged_plot(self):
        self.update_plot(self.canvas.axes_merged, self.im_merged)

    def update_all_plots(self):
        self.update_initial_plot()
        self.update_to_merge_plot()
        self.update_merged_plot()

    def compute_matches_and_affine_transformation(self):
        if (self.im_initial is None) or (self.im_to_merge is None):
            return

        akaze = cv2.AKAZE_create()
        kpts1, desc1 = akaze.detectAndCompute(self.im_initial, None)
        kpts2, desc2 = akaze.detectAndCompute(self.im_to_merge, None)

        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        nn_matches = matcher.knnMatch(desc1, desc2, 2)

        matched1 = []
        matched2 = []
        nn_match_ratio = 0.8  # Nearest neighbor matching ratio
        for m, n in nn_matches:
            if m.distance < nn_match_ratio * n.distance:
                matched1.append(kpts1[m.queryIdx])
                matched2.append(kpts2[m.trainIdx])

        matched1 = cv2.KeyPoint_convert(matched1)
        matched2 = cv2.KeyPoint_convert(matched2)

        self.affine, inliers = cv2.estimateAffine2D(matched1, matched2)
        inliers = inliers.flatten().astype(bool)
        self.matched1 = matched1[inliers]
        self.matched2 = matched2[inliers]

        self.draw_matches()
        print("matching results")
        print("number of matches = "+str(len(self.matched1)))
        print("affine transformation = "+str(self.affine))

    def draw_matches(self):
        if len(self.matched1) == 0:
            return

        self.canvas.axes_initial.plot(self.matched1[:, 0], self.matched1[:, 1], "*")
        self.canvas.axes_to_merge.plot(self.matched2[:, 0], self.matched2[:, 1], "*")

        for i in range(len(self.matched1)):
            xy_a = (self.matched1[i, 0], self.matched1[i, 1])
            xy_b = (self.matched2[i, 0], self.matched2[i, 1])
            con1 = ConnectionPatch(xyA=xy_a, xyB=xy_b, coordsA="data", coordsB="data",
                                   axesA=self.canvas.axes_initial, axesB=self.canvas.axes_to_merge, color="blue")
            self.canvas.axes_to_merge.add_artist(con1)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def merge_images(self):
        if self.affine is None:
            return

        im_to_merge_warped, im_merged_warped = warpAffinePadded(self.im_initial, self.im_to_merge, self.affine)
        is_not_black_im_merged = cv2.cvtColor(im_merged_warped, cv2.COLOR_RGB2GRAY) > 0
        is_not_black_im_to_merge_warped = cv2.cvtColor(im_to_merge_warped, cv2.COLOR_RGB2GRAY) > 0
        use_im_merged_warped = np.logical_and(np.logical_xor(is_not_black_im_to_merge_warped, is_not_black_im_merged), is_not_black_im_merged)

        self.im_merged = im_to_merge_warped
        self.im_merged[use_im_merged_warped] = im_merged_warped[use_im_merged_warped]

        self.update_merged_plot()

    def load_initial_image(self, file_name):
        if len(file_name) == 0:
            return
        self.im_initial = cv2.cvtColor(cv2.imread(file_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        self.update_initial_plot()

    def load_to_merge_image(self, file_name):
        if len(file_name) == 0:
            return
        self.im_to_merge = cv2.cvtColor(cv2.imread(file_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        self.update_to_merge_plot()

    def save_merged_image(self, file_path):
        if len(file_path) == 0:
            return
        cv2.imwrite(file_path, cv2.cvtColor(self.im_merged, cv2.COLOR_RGB2BGR))
        self.im_initial: typing.Union[cv2.Mat, None] = None
        self.im_to_merge: typing.Union[cv2.Mat, None] = None
        self.im_merged: typing.Union[cv2.Mat, None] = None
        self.affine: typing.Union[cv2.Mat, None] = None
        self.matched1: typing.Union[npt.NDArray, None] = None
        self.matched2: typing.Union[npt.NDArray, None] = None
        self.load_initial_image(file_path)
        self.update_all_plots()


def is_pixel_black(pixel_value):
    if len(pixel_value) == 3:
        return (pixel_value[0] == 0) and (pixel_value[1] == 0) and (pixel_value[2] == 0)
    else:
        return pixel_value == 0

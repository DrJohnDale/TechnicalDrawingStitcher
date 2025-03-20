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

        # progress bar settings
        self.compute_matches_and_affine_transformation_steps = 12
        self.merge_images_steps = 7
        self.save_merged_image_steps = 7

        # akaze settings
        self.pad = 25
        self.threshold = 0.01
        self.n_octaves = 1
        self.n_octaveLayers = 1
        self.min_selection_window_area = 100  # if the volume of a selection window is less than this is is ignored
        self.ransac_re_projection_threshold = 5
        #plotted points
        self.im_initial_plot_points = None
        self.im_to_merge_plot_points = None
        self.plot_point_lines = list()

    def update_plot(self, axes, im):
        axes.cla()
        if im is not None:
            axes.imshow(im)
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def update_initial_plot(self):
        self.clear_drawn_matches()
        self.update_plot(self.canvas.axes_initial, self.im_initial)
        self.canvas.set_rectangle_selector_initial()

    def update_to_merge_plot(self):
        self.clear_drawn_matches()
        self.update_plot(self.canvas.axes_to_merge, self.im_to_merge)
        self.canvas.set_rectangle_selector_to_merge()

    def update_merged_plot(self):
        self.update_plot(self.canvas.axes_merged, self.im_merged)

    def update_all_plots(self):
        self.update_initial_plot()
        self.update_to_merge_plot()
        self.update_merged_plot()

    def remove_and_shift_padded_keypoints(self, kpts, desc, rows: int, columns: int):
        selected_keypoints = list()
        to_select = list()
        for keypoint in kpts:
            new_pt = (keypoint.pt[0] - self.pad, keypoint.pt[1] - self.pad)
            keypoint.pt = new_pt
            if (0 <= new_pt[0] < columns) and (0 <= new_pt[1] < rows):
                selected_keypoints.append(keypoint)
                to_select.append(True)
            else:
                print(new_pt, rows, columns)
                to_select.append(False)
        selected_keypoints = tuple(selected_keypoints)
        return selected_keypoints, desc[to_select]

    def can_use_selection_rectangle(self, extents):
        dx = np.abs(extents[0] - extents[1])
        dy = np.abs(extents[2] - extents[3])
        area = np.sqrt(dx**2 + dy**2)
        return area >= self.min_selection_window_area

    def compute_matches_and_affine_transformation(self, progress_bar: QtWidgets.QProgressDialog):
        if (self.im_initial is None) or (self.im_to_merge is None):
            return

        akaze = cv2.AKAZE_create(threshold=self.threshold, nOctaves=self.n_octaves, nOctaveLayers=self.n_octaveLayers)

        progress_bar.setValue(1)

        im_initial_use_rect = self.can_use_selection_rectangle(self.canvas.rs_initial.extents)
        im_to_merge_use_rect = self.can_use_selection_rectangle(self.canvas.rs_to_merge.extents)
        print(im_initial_use_rect, im_to_merge_use_rect)
        if im_initial_use_rect:
            im_initial_select = self.im_initial[int(np.floor(self.canvas.rs_initial.extents[2])): int(np.floor(self.canvas.rs_initial.extents[3])),
                                                int(np.floor(self.canvas.rs_initial.extents[0])): int(np.floor(self.canvas.rs_initial.extents[1])), :]
        else:
            im_initial_select = self.im_initial
        progress_bar.setValue(2)

        if im_to_merge_use_rect:
            im_to_merge_select = self.im_to_merge[int(np.floor(self.canvas.rs_to_merge.extents[2])): int(np.floor(self.canvas.rs_to_merge.extents[3])),
                                                  int(np.floor(self.canvas.rs_to_merge.extents[0])): int(np.floor(self.canvas.rs_to_merge.extents[1])), :]
        else:
            im_to_merge_select = self.im_to_merge
        progress_bar.setValue(3)


        if self.pad > 0:
            im_initial_pad = np.pad(im_initial_select, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='constant', constant_values=0)
            im_to_merge_pad = np.pad(im_to_merge_select, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='constant', constant_values=0)
        else:
            im_initial_pad = im_initial_select
            im_to_merge_pad = im_to_merge_select
        progress_bar.setValue(4)
        kpts1, desc1 = akaze.detectAndCompute(im_initial_pad, None)
        kpts2, desc2 = akaze.detectAndCompute(im_to_merge_pad, None)
        progress_bar.setValue(5)

        if len(kpts1) == 0:
            print("no keypoints detected in im_initial")
            return

        if len(kpts2) == 0:
            print("no keypoints detected in im_to_merge")
            return

        # remove features in the pad area and shift the keypoints
        if self.pad > 0:
            kpts1, desc1 = self.remove_and_shift_padded_keypoints(kpts1, desc1, im_initial_select.shape[0], im_initial_select.shape[1])
            kpts2, desc2 = self.remove_and_shift_padded_keypoints(kpts2, desc2, im_to_merge_select.shape[0], im_to_merge_select.shape[1])
        progress_bar.setValue(6)

        # correct kpts for rectangle
        if im_initial_use_rect:
            x_0 = int(np.floor(self.canvas.rs_initial.extents[0]))
            y_0 = int(np.floor(self.canvas.rs_initial.extents[2]))
            for keypoint in kpts1:
                new_pt = (keypoint.pt[0] + x_0, keypoint.pt[1] + y_0)
                keypoint.pt = new_pt

        if im_to_merge_use_rect:
            x_0 = int(np.floor(self.canvas.rs_to_merge.extents[0]))
            y_0 = int(np.floor(self.canvas.rs_to_merge.extents[2]))
            for keypoint in kpts2:
                new_pt = (keypoint.pt[0] + x_0, keypoint.pt[1] + y_0)
                keypoint.pt = new_pt
        progress_bar.setValue(7)

        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        nn_matches = matcher.knnMatch(desc1, desc2, 2)
        progress_bar.setValue(8)

        matched1 = []
        matched2 = []
        nn_match_ratio = 0.8  # Nearest neighbor matching ratio
        for m, n in nn_matches:
            if m.distance < nn_match_ratio * n.distance:
                matched1.append(kpts1[m.queryIdx])
                matched2.append(kpts2[m.trainIdx])

        matched1 = cv2.KeyPoint_convert(matched1)
        matched2 = cv2.KeyPoint_convert(matched2)
        progress_bar.setValue(9)

        # self.affine, inliers = cv2.estimateAffinePartial2D(matched1, matched2)
        self.affine, inliers = cv2.estimateAffine2D(matched2, matched1, ransacReprojThreshold=self.ransac_re_projection_threshold)
        inliers = inliers.flatten().astype(bool)
        self.matched1 = matched1[inliers]
        self.matched2 = matched2[inliers]
        progress_bar.setValue(10)

        self.draw_matches()
        print("matching results")
        print("number of matches = "+str(len(self.matched1)))
        print("affine transformation = "+str(self.affine))
        progress_bar.setValue(11)

    def clear_drawn_matches(self):
        if self.im_initial_plot_points is not None:
            self.im_initial_plot_points[0].remove()
        if self.im_to_merge_plot_points is not None:
            self.im_to_merge_plot_points[0].remove()

        if len(self.plot_point_lines) > 0:
            for art in self.plot_point_lines:
                art.remove()

        self.plot_point_lines = list()
        self.im_initial_plot_points = None
        self.im_to_merge_plot_points = None

    def draw_matches(self):
        if len(self.matched1) == 0:
            return

        self.clear_drawn_matches()

        self.im_initial_plot_points = self.canvas.axes_initial.plot(self.matched1[:, 0], self.matched1[:, 1], "*")
        self.im_to_merge_plot_points = self.canvas.axes_to_merge.plot(self.matched2[:, 0], self.matched2[:, 1], "*")

        self.plot_point_lines = list()
        for i in range(len(self.matched1)):
            xy_a = (self.matched1[i, 0], self.matched1[i, 1])
            xy_b = (self.matched2[i, 0], self.matched2[i, 1])
            con1 = ConnectionPatch(xyA=xy_a, xyB=xy_b, coordsA="data", coordsB="data",
                                   axesA=self.canvas.axes_initial, axesB=self.canvas.axes_to_merge, color="blue")
            con_artist = self.canvas.axes_to_merge.add_artist(con1)
            self.plot_point_lines.append(con_artist)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def merge_images(self, progress_bar: QtWidgets.QProgressDialog):
        if self.affine is None:
            return
        progress_bar.setValue(1)
        im_merged_warped, im_to_merge_warped = warpAffinePadded(self.im_to_merge, self.im_initial, self.affine)
        progress_bar.setValue(2)
        is_not_black_im_merged = cv2.cvtColor(im_merged_warped, cv2.COLOR_RGB2GRAY) > 0
        progress_bar.setValue(3)
        is_not_black_im_to_merge_warped = cv2.cvtColor(im_to_merge_warped, cv2.COLOR_RGB2GRAY) > 0
        progress_bar.setValue(4)
        use_im_merged_warped = np.logical_and(np.logical_xor(is_not_black_im_to_merge_warped, is_not_black_im_merged), is_not_black_im_merged)
        progress_bar.setValue(5)

        self.im_merged = im_to_merge_warped
        self.im_merged[use_im_merged_warped] = im_merged_warped[use_im_merged_warped]
        progress_bar.setValue(6)

        self.update_merged_plot()
        progress_bar.setValue(6)

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

    def save_merged_image(self, file_path, progress_bar: QtWidgets.QProgressDialog):
        if len(file_path) == 0:
            return
        progress_bar.setValue(1)
        cv2.imwrite(file_path, cv2.cvtColor(self.im_merged, cv2.COLOR_RGB2BGR))
        progress_bar.setValue(2)
        self.im_initial = self.im_merged
        # self.im_initial: typing.Union[cv2.Mat, None] = None
        self.im_to_merge: typing.Union[cv2.Mat, None] = None
        self.im_merged: typing.Union[cv2.Mat, None] = None
        self.affine: typing.Union[cv2.Mat, None] = None
        self.matched1: typing.Union[npt.NDArray, None] = None
        self.matched2: typing.Union[npt.NDArray, None] = None
        progress_bar.setValue(3)
        # self.load_initial_image(file_path)
        progress_bar.setValue(4)
        self.clear_drawn_matches()
        progress_bar.setValue(5)
        self.update_all_plots()
        progress_bar.setValue(6)


def is_pixel_black(pixel_value):
    if len(pixel_value) == 3:
        return (pixel_value[0] == 0) and (pixel_value[1] == 0) and (pixel_value[2] == 0)
    else:
        return pixel_value == 0

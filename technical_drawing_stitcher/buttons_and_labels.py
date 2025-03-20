from PyQt5 import QtWidgets, QtCore, Qt
from technical_drawing_stitcher.core import Core


def get_buttons_and_labels_layout(core: Core):
    b_initial_image = QtWidgets.QPushButton("select initial image")
    b_initial_image.clicked.connect(lambda state: load_initial_callback(core))

    b_to_merge_image = QtWidgets.QPushButton("select to merge image")
    b_to_merge_image.clicked.connect(lambda state: load_to_merge_callback(core))

    b_compute_matches_and_affine_transformation = QtWidgets.QPushButton("compute matches")
    b_compute_matches_and_affine_transformation.clicked.connect(lambda state: compute_matches_and_affine_transformation(core))

    b_merge_images = QtWidgets.QPushButton("merge images")
    b_merge_images.clicked.connect(lambda state: merge_images(core))

    b_save_merged_image = QtWidgets.QPushButton("save merged image")
    b_save_merged_image.clicked.connect(lambda state: save_merged_callback(core))

    layout_buttons = QtWidgets.QVBoxLayout()
    layout_buttons.addWidget(b_initial_image)
    layout_buttons.addWidget(b_to_merge_image)
    layout_buttons.addWidget(b_compute_matches_and_affine_transformation)
    layout_buttons.addWidget(b_merge_images)
    layout_buttons.addWidget(b_save_merged_image)
    return layout_buttons


def load_initial_callback(core: Core):
    file_name, _ = QtWidgets.QFileDialog.getOpenFileName(core.main_window, "select initial image", "","")
    print(file_name)
    core.load_initial_image(file_name)


def load_to_merge_callback(core: Core):
    file_name, _ = QtWidgets.QFileDialog.getOpenFileName(core.main_window, "select to merge image", "","")
    print(file_name)
    core.load_to_merge_image(file_name)


def save_merged_callback(core: Core):
    file_name, _ = QtWidgets.QFileDialog.getSaveFileName(core.main_window, "select save to merge image", "", "")
    progress = QtWidgets.QProgressDialog("Saving Merged", "", 0, core.save_merged_image_steps)
    progress.setMinimumDuration(0)
    progress.setModal(True)
    progress.setValue(0)
    print(file_name)
    core.save_merged_image(file_name, progress)
    progress.setValue(core.save_merged_image_steps)

def compute_matches_and_affine_transformation(core: Core):
    progress = QtWidgets.QProgressDialog("Computing matches and affine transformation","", 0, core.compute_matches_and_affine_transformation_steps)
    progress.setMinimumDuration(0)
    progress.setModal(True)
    progress.setValue(0)
    core.compute_matches_and_affine_transformation(progress)
    progress.setValue(core.compute_matches_and_affine_transformation_steps)

def merge_images(core: Core):
    progress = QtWidgets.QProgressDialog("Merging Images", "", 0, core.merge_images_steps)
    progress.setMinimumDuration(0)
    progress.setModal(True)
    progress.setValue(0)
    core.merge_images(progress)
    progress.setValue(core.merge_images_steps)

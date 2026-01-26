"""
Module that implements the UI widget of the algorithm.
"""
from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_uniface_parsing.infer_uniface_parsing_process import InferUnifaceParsingParam

# PyQt GUI framework
from PyQt5.QtWidgets import *


class InferUnifaceParsingWidget(core.CWorkflowTaskWidget):
    """
    Class that implements UI widget to adjust algorithm parameters.
    Inherits PyCore.CWorkflowTaskWidget from Ikomia API.
    """

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferUnifaceParsingParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Model name combo box
        self.combo_model = pyqtutils.append_combo(
            self.grid_layout, "Model name")
        self.combo_model.addItem("resnet18")
        self.combo_model.addItem("resnet34")
        self.combo_model.setCurrentText(self.parameters.model_name)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        """QT slot called when users click the Apply button."""
        # Get parameters from widget
        self.parameters.model_name = self.combo_model.currentText()

        # Send signal to launch the algorithm main function
        self.emit_apply(self.parameters)


class InferUnifaceParsingWidgetFactory(dataprocess.CWidgetFactory):
    """
    Factory class to create algorithm widget object.
    Inherits PyDataProcess.CWidgetFactory from Ikomia API.
    """

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_uniface_parsing"

    def create(self, param):
        """Instantiate widget object."""
        return InferUnifaceParsingWidget(param, None)

"""
Main Ikomia plugin module.
Ikomia Studio and Ikomia API use it to load algorithms dynamically.
"""
from ikomia import dataprocess
from infer_uniface_parsing.infer_uniface_parsing_process import InferUnifaceParsingFactory
from infer_uniface_parsing.infer_uniface_parsing_process import InferUnifaceParsingParamFactory


class IkomiaPlugin(dataprocess.CPluginProcessInterface):
    """
    Interface class to integrate the process with Ikomia application.
    Inherits PyDataProcess.CPluginProcessInterface from Ikomia API.
    """
    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        """Instantiate process object."""
        return InferUnifaceParsingFactory()

    def get_widget_factory(self):
        """Instantiate associated widget object."""
        from infer_uniface_parsing.infer_uniface_parsing_widget import InferUnifaceParsingWidgetFactory
        return InferUnifaceParsingWidgetFactory()

    def get_param_factory(self):
        """Instantiate algorithm parameters object."""
        return InferUnifaceParsingParamFactory()

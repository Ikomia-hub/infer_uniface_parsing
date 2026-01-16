"""
Module that implements the core logic of algorithm execution.
"""
import copy
import os
import cv2
from ikomia import core, dataprocess
from uniface.visualization import vis_parsing_maps
from .models.model_loader import create_parser


class InferUnifaceParsingParam(core.CWorkflowTaskParam):
    """
    Class to handle the algorithm parameters.
    Inherits PyCore.CWorkflowTaskParam from Ikomia API.
    """

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Model options: "resnet18" (faster) or "resnet34" (more accurate)
        self.model_name = "resnet18"
        self.update = False

    def set_values(self, params):
        """
        Set parameters values from Ikomia Studio or API.
        Parameters values are stored as string and accessible like a python dict.
        """
        self.model_name = params.get("model_name", "resnet18")
        self.update = True

    def get_values(self):
        """
        Send parameters values to Ikomia Studio or API.
        Create the specific dict structure (key-value as string).
        """
        params = {
            "model_name": str(self.model_name)
        }
        return params


class InferUnifaceParsingParamFactory(dataprocess.CTaskParamFactory):
    """Factory class to create parameters object."""

    def __init__(self):
        dataprocess.CTaskParamFactory.__init__(self)
        self.name = "infer_uniface_parsing"

    def create(self):
        """Instantiate parameters object."""
        return InferUnifaceParsingParam()


class InferUnifaceParsing(dataprocess.CSemanticSegmentationTask):
    """
    Class that implements the algorithm.
    Inherits PyCore.CWorkflowTask or derived from Ikomia API.
    """

    def __init__(self, name, param):
        dataprocess.CSemanticSegmentationTask.__init__(self, name)
        # Add input/output of the algorithm here
        # Input: Image
        # Output 0: Semantic segmentation (from parent class)
        # Output 1: Visualization image overlay
        self.add_output(dataprocess.CImageIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferUnifaceParsingParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        # Class names for face parsing (19 classes)
        self.class_names = [
            "background",     # 0
            "skin",          # 1
            "left_eyebrow",  # 2
            "right_eyebrow",  # 3
            "left_eye",      # 4
            "right_eye",     # 5
            "eye_glasses",   # 6
            "left_ear",      # 7
            "right_ear",     # 8
            "ear_ring",      # 9
            "nose",          # 10
            "mouth",         # 11
            "upper_lip",     # 12
            "lower_lip",     # 13
            "neck",          # 14
            "neck_lace",     # 15
            "cloth",         # 16
            "hair",          # 17
            "hat"            # 18
        ]

        self.model_folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "weights")
        self.parser = None

    def _load_model(self):
        """Load the BiSeNet parser model using the refactored model loader."""
        param = self.get_param_object()

        # Create parser with model weights saved in self.model_folder
        self.parser = create_parser(
            model_name=param.model_name,
            model_folder=self.model_folder
        )

    def init_long_process(self):
        """Initialize model for long processing."""
        self._load_model()
        super().init_long_process()

    def get_progress_steps(self):
        """
        Ikomia Studio only.
        Function returning the number of progress steps for this algorithm.
        This is handled by the main progress bar of Ikomia Studio.
        """
        return 1

    def run(self):
        """Main function and entry point for algorithm execution."""
        # Call begin_task_run() for initialization
        self.begin_task_run()
        param = self.get_param_object()

        # Get input image
        img_input = self.get_input(0)
        src_image = img_input.get_image()

        # Convert RGBA to RGB if necessary
        if src_image.shape[-1] == 4:  # RGBA format
            src_image = src_image[:, :, :3]  # Keep only RGB channels

        # Load model if needed
        if self.parser is None or param.update:
            self._load_model()
            param.update = False

        # Parse the face to get segmentation mask
        mask = self.parser.parse(src_image)

        if mask is None:
            self.end_task_run()
            return

        # Set class names for semantic segmentation output
        self.set_names(self.class_names)

        # Set the segmentation mask to output 0 (from parent CSemanticSegmentationTask)
        self.set_mask(mask)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


class InferUnifaceParsingFactory(dataprocess.CTaskFactory):
    """
    Factory class to create process object.
    Inherits PyDataProcess.CTaskFactory from Ikomia API.
    """

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_uniface_parsing"
        self.info.short_description = "Face parsing (semantic segmentation) using UniFace BiSeNet model."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Yakhyokhuja Valikhujaev"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2024
        self.info.license = "MIT License"

        # Ikomia API compatibility
        self.info.min_ikomia_version = "0.15.0"

        # URL of documentation
        self.info.documentation_link = "https://yakhyo.github.io/uniface/"

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_uniface_parsing"
        self.info.original_repository = "https://github.com/yakhyo/uniface"

        # Keywords used for search
        self.info.keywords = "uniface, parsing, face parsing, semantic segmentation, BiSeNet"

        # General type: INFER, TRAIN, DATASET or OTHER
        self.info.algo_type = core.AlgoType.INFER

        # Algorithms tasks
        self.info.algo_tasks = "SEMANTIC_SEGMENTATION"

        # Hardware configuration
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 8
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 4

    def create(self, param=None):
        """Instantiate algorithm object."""
        return InferUnifaceParsing(self.info.name, param)

"""
Model loader with monkey-patching to save weights in custom folder.
"""
import os
from typing import Any

from uniface.model_store import verify_model_weights as original_verify_model_weights


class ModelLoader:
    """
    Model loader that patches verify_model_weights to save models in a custom folder.
    """

    def __init__(self, model_folder: str):
        """
        Initialize the model loader.

        Args:
            model_folder: Directory where model weights should be saved.
        """
        self.model_folder = model_folder
        self._patched_modules = []

    def _patch_verify_model_weights(self):
        """Monkey-patch verify_model_weights to use self.model_folder."""
        def patched_verify_model_weights(model_name, root=None):
            if root is None:
                root = self.model_folder
            return original_verify_model_weights(model_name, root=root)

        # Patch in the model_store module
        import uniface.model_store
        original_func_module = uniface.model_store.verify_model_weights
        uniface.model_store.verify_model_weights = patched_verify_model_weights
        self._patched_modules.append(
            ('uniface.model_store', original_func_module))

        # Patch in detection modules that import it directly
        detection_modules = ['retinaface', 'yolov5', 'scrfd', 'yolov8']
        for module_name in detection_modules:
            try:
                module = __import__(
                    f'uniface.detection.{module_name}', fromlist=[module_name])
                if hasattr(module, 'verify_model_weights'):
                    original_func = module.verify_model_weights
                    module.verify_model_weights = patched_verify_model_weights
                    self._patched_modules.append(
                        (f'uniface.detection.{module_name}', original_func))
            except (ImportError, AttributeError):
                # Module doesn't exist or doesn't have verify_model_weights, skip
                pass

        # Patch in parsing modules that import it directly
        try:
            import uniface.parsing.bisenet
            if hasattr(uniface.parsing.bisenet, 'verify_model_weights'):
                original_func = uniface.parsing.bisenet.verify_model_weights
                uniface.parsing.bisenet.verify_model_weights = patched_verify_model_weights
                self._patched_modules.append(
                    ('uniface.parsing.bisenet', original_func))
        except (ImportError, AttributeError):
            pass

    def _restore_original_functions(self):
        """Restore original verify_model_weights functions."""
        for module_path, original_func in self._patched_modules:
            parts = module_path.split('.')
            module = __import__(module_path, fromlist=[parts[-1]])
            module.verify_model_weights = original_func
        self._patched_modules.clear()

    def create_detector(self, model_type: str, **kwargs: Any):
        """
        Create a detector instance with patched model loading.

        Args:
            model_type: Type of detector ('retinaface', 'yolov5face', 'scrfd', or 'yolov8face').
            **kwargs: Additional arguments to pass to the detector constructor.

        Returns:
            Detector instance.
        """
        # Ensure model folder exists
        os.makedirs(self.model_folder, exist_ok=True)

        # Patch the verify_model_weights function BEFORE importing detector classes
        # This is critical because detector modules import verify_model_weights at module level
        self._patch_verify_model_weights()

        try:
            # Import detector classes after patching
            # The patched verify_model_weights will be used when the detector is instantiated
            if model_type.lower() == 'retinaface':
                from uniface.detection import RetinaFace
                return RetinaFace(**kwargs)
            elif model_type.lower() == 'yolov5face':
                from uniface.detection import YOLOv5Face
                return YOLOv5Face(**kwargs)
            elif model_type.lower() == 'scrfd':
                from uniface.detection import SCRFD
                return SCRFD(**kwargs)
            elif model_type.lower() == 'yolov8face':
                from uniface.detection import YOLOv8Face
                return YOLOv8Face(**kwargs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}. "
                                 f"Supported types: retinaface, yolov5face, scrfd, yolov8face")
        finally:
            # Restore original functions
            self._restore_original_functions()

    def create_parser(self, model_name: str = "resnet18"):
        """
        Create a BiSeNet parser instance with patched model loading.

        Args:
            model_name: Model name ('resnet18' or 'resnet34').

        Returns:
            BiSeNet parser instance.
        """
        # Ensure model folder exists
        os.makedirs(self.model_folder, exist_ok=True)

        # Patch the verify_model_weights function BEFORE importing parser
        self._patch_verify_model_weights()

        try:
            from uniface.parsing import BiSeNet
            from uniface.constants import ParsingWeights

            # Map model name to weights enum
            if model_name.lower() == "resnet18":
                weights = ParsingWeights.RESNET18
            elif model_name.lower() == "resnet34":
                weights = ParsingWeights.RESNET34
            else:
                raise ValueError(f"Unsupported model name: {model_name}. "
                                 f"Supported: resnet18, resnet34")

            return BiSeNet(model_name=weights)
        finally:
            # Restore original functions
            self._restore_original_functions()


def create_detector(model_type: str, model_folder: str, **kwargs: Any):
    """
    Convenience function to create a detector with custom model folder.

    Args:
        model_type: Type of detector ('retinaface' or 'yolov5face').
        model_folder: Directory where model weights should be saved.
        **kwargs: Additional arguments to pass to the detector constructor.

    Returns:
        Detector instance.
    """
    loader = ModelLoader(model_folder)
    return loader.create_detector(model_type, **kwargs)


def create_parser(model_name: str, model_folder: str):
    """
    Convenience function to create a BiSeNet parser with custom model folder.

    Args:
        model_name: Model name ('resnet18' or 'resnet34').
        model_folder: Directory where model weights should be saved.

    Returns:
        BiSeNet parser instance.
    """
    loader = ModelLoader(model_folder)
    return loader.create_parser(model_name)

import inspect
import pkgutil


class Processor:
    """Base class that each processor must inherit from."""

    def __init__(
        self,
        options=None,
        relative_dir=None,
        image_instance_ops=None,
    ):
        self.options = options
        self.relative_dir = relative_dir
        self.image_instance_ops = image_instance_ops
        self.tuning_config = image_instance_ops.tuning_config
        self.description = "UNKNOWN"


class ProcessorManager:
    """Upon creation, this class will read the processors package for modules
    that contain a class definition that is inheriting from the Processor class
    """

    def __init__(self, processors_dir="src.processors"):
        """Constructor that initiates the reading of all available processors
        when an instance of the ProcessorCollection object is created
        """
        self.processors_dir = processors_dir

    @staticmethod
    def get_name_filter(processor_name):
        def filter_function(member):
            return inspect.isclass(member) and member.__module__ == processor_name

        return filter_function


# Singleton export
PROCESSOR_MANAGER = ProcessorManager()

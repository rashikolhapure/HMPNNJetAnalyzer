import os

from hep.config import Paths
from .genutils import (
    check_dir,
)


class Method(object):
    """
    Base class for methods.

    Args:
        *args: Variable-length positional arguments.
        **kwargs: Variable-length keyword arguments.

    Attributes:
        input_data: The input data for the method.
        output_data: The output data for the method.
        max_count: Maximum count (optional, default is None).
        count: Current count (initialized to 0).
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a Method instance.

        Args:
            *args: Variable-length positional arguments.
            **kwargs: Variable-length keyword arguments.
        """
        compulsory_kwargs = (
            "input_data",
            "output_data",
        )
        self.input_data = kwargs.get("input_data")
        self.output_data = kwargs.get("output_data")
        self.max_count = None
        self.count = 0


class PhysicsMethod(Method):
    """
    Base class for physics methods.

    Args:
        *args: Variable-length positional arguments.
        **kwargs: Variable-length keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a PhysicsMethod instance.

        Args:
            *args: Variable-length positional arguments.
            **kwargs: Variable-length keyword arguments.
        """
        super().__init__(args, **kwargs)

    def __iter__(self):
        """
        Return an iterator for the PhysicsMethod instance.
        """
        return self

    def __len__(self):
        """
        Get the length of the PhysicsMethod instance.

        Returns:
            int: The length of the PhysicsMethod instance.
        """
        assert self.max_count, "Calling uninitialized " + type(self).__name__
        return self.max_count


class NetworkMethod(Method):
    """
    Base class for network methods.

    Args:
        *args: Variable-length positional arguments.
        **kwargs: Variable-length keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Data(object):
    """
    Base class for data handling.

    Args:
        *args: Variable-length positional arguments.
        **kwargs: Variable-length keyword arguments.

    Attributes:
        dtypes: Data types.
        prefix_path: Prefix path (optional).
        reader_method: Reader method.
        writer_method: Writer method.
        data_ext: Data extension.
        file_ext: File extension (initialized to None).
        mg_event_path: MadGraph event path.
        max_count: Maximum count (initialized to "NA").
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a Data instance.

        Args:
            *args: Variable-length positional arguments.
            **kwargs: Variable-length keyword arguments.
        """
        compulsory_keys = {
            "reader_method",
            "writer_method",
            "run_name",
        }
        # assert compulsory_keys.issubset(set(kwargs.keys()))
        self.dtypes = args
        self.prefix_path = kwargs.get("prefix_path")
        self.reader_method = kwargs.get("reader_method")
        self.writer_method = kwargs.get("writer_method")
        self.data_ext = kwargs.get("extension")
        self.file_ext = None
        self.mg_event_path = os.path.join(
            Paths.madgraph_dir,
            kwargs["run_name"],
            "Events",
        )
        self.max_count = "NA"


class PhysicsData(Data):
    """
    Base class for physics data handling.

    Args:
        *args: Variable-length positional arguments.
        **kwargs: Variable-length keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a PhysicsData instance.

        Args:
            *args: Variable-length positional arguments.
            **kwargs: Variable-length keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """
        Return an iterator for the PhysicsData instance.
        """
        return self

    def __len__(self):
        """
        Get the length of the PhysicsData instance.

        Returns:
            str: The maximum count of the PhysicsData instance (initialized to "NA").
        """
        return self.max_count
    
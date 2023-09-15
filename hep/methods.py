from ..classes import (
    PhysicsMethod,
)
from ..io.saver import (
    Pickle,
    Unpickle,
)
from ..genutils import (
    print_events,
    pool_splitter,
)
from .config import (
    FinalStates,
    Paths,
    EventAttribute,
)
from .data import (
    RootEvents,
    NumpyEvents,
    PassedEvents,
    PreProcessedEvents,
)
import sys
import os

import numpy as np

np.set_printoptions(precision=16)


class OverwriteError(Exception):
    def __init__(
        self,
        message="OverwriteException",
    ):
        super(
            OverwriteError,
            self,
        ).__init__(message)

    class DelphesNumpy(PhysicsMethod):
        """
        Initialize a DelphesNumpy instance.

        Parameters:
        -----------
        run_name : str
            The name of the run.
        *args : positional arguments
            Positional arguments to pass to the parent class (PhysicsMethod).
        **kwargs : keyword arguments
            Keyword arguments to configure the DelphesNumpy instance.

        """

        def __init__(
            self,
            run_name,
            *args,
            **kwargs,
        ):
            print(
                self.__class__.__name__,
                run_name,
                args,
                kwargs,
            )
            super().__init__(
                *args,
                input_data="RootEvents",
                output_data="NumpyEvents",
            )
            self.run_name = run_name
            if kwargs.get("root_file_path") is None:
                self.in_data = RootEvents(
                    run_name,
                    return_vals=(
                        "Delphes",
                        "path",
                        "index",
                    ),
                    select_runs=kwargs.get(
                        "select_runs",
                        [],
                    ),
                    tag=kwargs.get(
                        "root_file_tag",
                        "",
                    ),
                    ignore_runs=kwargs.get(
                        "ignore_runs",
                        [],
                    ),
                )
            else:
                self.in_data = RootEvents(
                    run_name,
                    return_vals=(
                        "Delphes",
                        "path",
                        "index",
                    ),
                    root_file_path=kwargs.get("root_file_path"),
                )
            self.max_count = len(self.in_data)
            self.out_data = NumpyEvents(
                run_name,
                mode="w",
                max_count=self.in_data.max_count,
                prefix=kwargs.get(
                    "root_file_tag",
                    "",
                ),
            )
            if "root_file_path" in kwargs:
                self.out_data.mg_event_path = self.in_data.mg_event_path
            self.final_state_attributes = kwargs.get(
                "final_state_atributes",
                FinalStates.attributes,
            )
            if not kwargs.get(
                "include_gen_particles",
                False,
            ):
                print("Removing GenParticle from final state list...")
                self.extract_gen_particles = False
                self.final_state_attributes.pop("Particle")
            else:
                self.extract_gen_particles = True
            self.include_eflow = kwargs.get(
                "include_eflow",
                False,
            )
            if not self.include_eflow:
                try:
                    self.final_state_attributes.pop("EFlow")
                except KeyError:
                    pass
            self.include_track = kwargs.get(
                "include_track",
                False,
            )
            if not self.include_track:
                try:
                    self.final_state_attributes.pop("Track")
                except KeyError:
                    pass
            self.overwrite = kwargs.get(
                "overwrite",
                False,
            )
            self.root_tag = kwargs.get(
                "root_file_tag",
                "",
            )
            self.Events = None
            self.current = None
            print("Extracting final state attributes:\n")
            for (
                key,
                val,
            ) in self.final_state_attributes.items():
                print(f"{key:15}  {val}\n")

        def __next__(self):
            if self.count < self.max_count:
                self.count += 1
                self.current = next(self.in_data)
                try:
                    if self.overwrite:
                        raise OverwriteError
                    current = Unpickle(
                        self.out_data.prefix + "_delphes.pickle",
                        load_path=self.current["path"],
                    )
                    self.out_data.exception = False
                except Exception as e:
                    if "Particle" in self.final_state_attributes:
                        print(
                            "Excluding Particle for writing to pickle file..."
                        )
                        particle_atttributes = self.final_state_attributes.pop(
                            "Particle"
                        )
                    self.out_data.exception = True
                    print(e)
                    print(
                        "\nReading root file from ",
                        self.current["path"],
                    )
                    self.Events = self.current["Delphes"]
                    current = {}
                    for final_state in self.final_state_attributes:
                        # if self.exclude_gen_particles and
                        # final_State=="Particle": continue
                        print(
                            "Extracting branch :",
                            final_state,
                            "\nAttributes:",
                            self.final_state_attributes[final_state],
                        )
                        current[final_state] = self.get(
                            final_state,
                            self.final_state_attributes[final_state],
                        )
                    print("Adding EventAttribute...")
                    length = len(current[final_state])
                    event_attribute = [
                        EventAttribute(
                            run_name=self.run_name,
                            tag=self.root_tag,
                            path=self.current["path"],
                            index=_,
                        )
                        for _ in range(length)
                    ]
                    current["EventAttribute"] = event_attribute
                    print_events(current)
                    if (
                        "Particle" not in self.final_state_attributes
                        and self.extract_gen_particles
                    ):
                        print("Adding Particle class...")
                        self.final_state_attributes[
                            "Particle"
                        ] = particle_atttributes
                finally:
                    self.out_data.current_events = current
                    self.out_data.current_run = self.current["path"]
                    next(self.out_data)
                    if self.extract_gen_particles:
                        print(
                            "Extracting Particle : ",
                            self.current["path"],
                            "\nAttributes: ",
                            self.final_state_attributes["Particle"],
                        )
                        self.Events = self.current["Delphes"]
                        current["Particle"] = self.get(
                            "Particle",
                            self.final_state_attributes["Particle"],
                        )
                    return current
            else:
                raise StopIteration

    def get(
        self,
        final_state,
        attributes,
        indices=None,
    ):
        """
        Retrieve data from a specific final state with specified attributes.

        class method to directly select <final_state> with list of
        <attributes> at <indices> from <root_file>. returns a numpy array
        of either len(indices) with variable shape depending on the
        <final_state>.

        Parameters:
        -----------
        final_state : str
            The final state to retrieve data from.
        attributes : list of str
            A list of attribute names to retrieve for each event in the
            final state.
        indices : numpy.ndarray or None, optional
            An array of indices to select specific events. If None,
            all events are selected.

        Returns:
        --------
        numpy.ndarray
            An array containing the selected data with variable shape
            depending on the specified attributes."""
        (
            return_dict,
            temp_dict,
        ) = (
            dict(),
            dict(),
        )
        for item in attributes:
            temp_dict[item] = self.Events[final_state][
                final_state + "." + item
            ].array()
        if indices is None:
            indices = np.arange(len(temp_dict[item]))
        return_array = []
        for event_index in indices:
            array = [[] for i in range(len(attributes))]
            for i in range(len(attributes)):
                array[i] = np.array(
                    temp_dict[attributes[i]][event_index],
                    dtype="float64",
                )
            return_array.append(
                np.swapaxes(
                    np.array(
                        array,
                        dtype="float64",
                    ),
                    0,
                    1,
                )
            )
        return np.array(return_array)


class BaselineCuts(PhysicsMethod):
    def __init__(
        self,
        run_name,
        *args,
        **kwargs,
    ):
        """
        Initialize a BaselineCuts instance.

        Parameters:
        -----------
        run_name : str
            The name of the run.
        *args : positional arguments
            Positional arguments to pass to the parent class (PhysicsMethod).
        **kwargs : keyword arguments
            Keyword arguments to configure the BaselineCuts instance.

        """
        super().__init__(
            input_data="NumpyEvents",
            output_data="PassedEvents",
        )
        self.num_cores = kwargs.get("num_cores")
        self.read_partons = kwargs.get(
            "read_partons",
            None,
        )
        self.exclude_keys = kwargs.get(
            "exclude_keys",
            [],
        )
        self.ignore_keys = kwargs.get("ignore_keys", [])
        self.run_name = run_name
        self.in_data = NumpyEvents(
            run_name,
            mode="r",
            run_tag=kwargs.get(
                "run_tag",
                "None",
            ),
            prefix=kwargs.get(
                "delphes_prefix",
                "",
            ),
            both_dirs=kwargs.get(
                "both_dirs",
                False,
            ),
            select_runs=kwargs.get(
                "select_runs",
                [],
            ),
            ignore_runs=kwargs.get(
                "ignore_runs",
                [],
            ),
        )
        self.out_data = PassedEvents(
            run_name,
            mode="w",
            max_count=self.in_data.max_count,
            tag=kwargs.get("tag", ""),
            save=kwargs.get("save", False),
        )
        self.count = 0
        self.cut_args = kwargs.get(
            "cut_args",
            {},
        )
        self.max_count = self.in_data.max_count
        self.cut = None

    def __next__(self):
        assert self.cut, "Set a cut first"
        if self.count < self.max_count:
            self.count += 1
            in_events = next(self.in_data)
            if self.read_partons is not None:
                print("Adding parton level information!")
                parton_dict = self.read_partons(
                    path=self.in_data.current_run,
                    run_name=self.run_name,
                )
                print_events(
                    parton_dict,
                    name="partonic events",
                )
                for (
                    key,
                    val,
                ) in parton_dict.items():
                    in_events["partonic_" + key] = val
            if "debug" in sys.argv:
                passed_events = self.cut(
                    in_events,
                    **self.cut_args,
                )
            else:
                if self.num_cores is not None:
                    passed_events = pool_splitter(
                        self.cut,
                        in_events,
                        exclude=self.exclude_keys,
                        ignore_keys=self.ignore_keys,
                        num_cores=self.num_cores,
                    )
                else:
                    passed_events = pool_splitter(
                        self.cut,
                        in_events,
                        ignore_keys=self.ignore_keys,
                        exclude=self.exclude_keys,
                    )
            self.out_data.current_run = self.in_data.current_run
            self.out_data.current_events = passed_events
            next(self.out_data)
            return passed_events
        else:
            raise StopIteration


class PreProcess(PhysicsMethod):
    def __init__(
        self,
        run_name,
        *args,
        **kwargs,
    ):
        """
        Initialize a PreProcess instance.

        Parameters:
        -----------
        run_name : str
            The name of the run.
        *args : positional arguments
            Positional arguments to pass to the parent class (PhysicsMethod).
        **kwargs : keyword arguments
            Keyword arguments to configure the PreProcess instance.

        """
        super().__init__(
            run_name=run_name,
            input_data="PassedEvents",
            output_data="PreprocessedEvents",
        )
        self.num_cores = kwargs.get("num_cores")
        self.in_data = PassedEvents(
            run_name,
            tag=kwargs.get(
                "cut_tag",
                "try",
            ),
            mode="r",
            run_tag=kwargs.get(
                "run_tag",
                "None",
            ),
            both_dirs=kwargs.get(
                "both_dirs",
                False,
            ),
            remove_keys=kwargs.get(
                "remove_keys",
                [],
            ),
        )
        self.max_count = len(self.in_data)
        self.out_data = PreProcessedEvents(
            run_name,
            tag=kwargs.get("tag", "try"),
            mode="w",
            max_count=self.max_count,
        )
        self.preprocess = None

    def __next__(self):
        assert self.preprocess, "Set preprocess function first!"
        if self.count < self.max_count:
            self.count += 1
            passed_events = next(self.in_data)
            if "debug" not in sys.argv:
                if self.num_cores is not None:
                    preprocessed = pool_splitter(
                        self.preprocess,
                        passed_events,
                        num_cores=self.num_cores,
                    )
                else:
                    preprocessed = pool_splitter(
                        self.preprocess,
                        passed_events,
                    )
            else:
                preprocessed = self.preprocess(passed_events)
            self.out_data.current_run = self.in_data.current_run
            self.out_data.current_events = preprocessed
            next(self.out_data)
            return preprocessed
        else:
            raise StopIteration


if __name__ == "__main__":
    now = DelphesNumpy(
        "wwx",
        tag="try",
        reconstructed_objects={
            "FatJet": 2,
            "Tower": None,
        },
        image_shapes={
            "FatJet": (
                32,
                32,
            ),
            "Tower": (
                100,
                100,
            ),
        },
    )
    # now.cut=now.sample_cut
    for (
        events,
        path,
        index,
    ) in now:
        print(path, index)
        print_events(events)

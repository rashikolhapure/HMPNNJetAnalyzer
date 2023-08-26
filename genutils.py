import os
import multiprocessing
import sys
from optparse import (
    OptionParser,
)

from .hep.config import (
    gen_particle_index,
    particle_to_PID,
    FinalStates,
    PID_to_particle,
)
import pickle
import numpy as np


def print_vectors(
    vectors, format="lorentz"
):
    """prints the array of vectors either in (pt,eta,phi,mass) or (px,py,pz,E)"""

    def print_single(vector):
        if (
            format
            == "lorentz"
        ):
            print(
                f"Px: {vector.px:10.8f}  Py: {vector.py:10.8f}  Pz: {vector.pz:10.8f}  E: {vector.e:10.8f}"
            )
        else:
            print(
                f"Pt: {vector.pt:.8f} Eta: {vector.eta:.8f} Phi: {vector.phi:.8f} Mass: {vector.mass:.8f}"
            )

    if hasattr(
        vectors, "__iter__"
    ):
        list(
            map(
                print_single,
                vectors,
            )
        )
    else:
        print_single(vectors)
    return


def rescale(array):
    print(array.shape)
    print(
        "Rescaling:\n mean values:",
        np.mean(
            array, axis=0
        ),
        " Standard Deviations: ",
        np.std(
            array, axis=0
        ),
    )
    array = (
        array
        - np.mean(
            array, axis=0
        )
    ) / np.std(array, axis=0)
    print(
        "Rescaling:\n mean values:",
        np.mean(
            array, axis=0
        ),
        " Standard Deviations: ",
        np.std(
            array, axis=0
        ),
    )
    return array


def print_particle(
    particle,
    four_vec=False,
    ind=None,
    mass=True,
):
    order = FinalStates.attributes[
        "Particle"
    ]
    start, stop = 4, -1
    # print (order,particle)
    if ind is not None:
        print(
            "Ind: ",
            ind,
            " ",
            end="",
        )
    [
        print(
            name,
            ": ",
            item,
            "  ",
            end="",
        )
        for name, item in zip(
            order[
                start:stop
            ],
            particle[
                start:stop
            ],
        )
    ]
    try:
        print(
            "Particle name:",
            PID_to_particle[
                particle[
                    gen_particle_index.PID
                ]
            ],
        )
    except KeyError:
        print(
            "Particle name not in SM particles"
        )
    if four_vec:
        print(
            "4-vec:", end=""
        )
        [
            print(
                "  ",
                round(
                    item, 4
                ),
                end="",
            )
            for item in particle[
                :4
            ]
        ]
        print()
    if mass:
        print(
            "Mass: ",
            particle[
                gen_particle_index.Mass
            ],
        )
    return 0


def choose_bin(
    events,
    bin_var,
    Range,
    var_key="Jet",
    **kwargs,
):
    print(
        "Chossing events in range: ",
        Range,
        " for variable: ",
        bin_var,
    )
    jets = events[var_key]
    if var_key == "FatJet":
        print(
            "Binning to be performed with FatJet!\n"
        )
        input(
            "Press enter to sum Fatjet constituents, or Ctrl + X to abort: "
        )
        jets = np.expand_dims(
            np.array(
                [
                    np.sum(
                        item
                    )
                    for item in jets
                ]
            ),
            -1,
        )
    print(Range, jets.shape)
    if bin_var == "pt_j0":
        true_indices = [
            i
            for i in range(
                len(jets)
            )
            if jets[i][
                0
            ].Pt()
            >= Range[0]
            and jets[i][
                0
            ].Pt()
            <= Range[1]
        ]
    elif bin_var == "pt_j1":
        true_indices = [
            i
            for i in range(
                len(jets)
            )
            if jets[i][
                1
            ].Pt()
            >= Range[0]
            and jets[i][
                1
            ].Pt()
            <= Range[1]
        ]
    elif bin_var == "eta_j0":
        true_indices = [
            i
            for i in range(
                len(jets)
            )
            if jets[i][
                0
            ].Eta()
            >= Range[0]
            and jets[i][
                0
            ].Eta()
            <= Range[1]
        ]
    elif bin_var == "eta_j1":
        true_indices = [
            i
            for i in range(
                len(jets)
            )
            if jets[i][
                1
            ].Eta()
            >= Range[0]
            and jets[i][
                1
            ].Eta()
            <= Range[1]
        ]
    elif (
        bin_var
        == "delta_eta"
    ):
        true_indices = [
            i
            for i in range(
                len(jets)
            )
            if abs(
                jets[i][
                    0
                ].Eta()
                - jets[i][
                    1
                ].Eta()
            )
            >= Range[0]
            and abs(
                jets[i][
                    0
                ].Eta()
                - jets[i][
                    1
                ].Eta()
            )
            <= Range[1]
        ]
    elif (
        bin_var
        == "delta_phi"
    ):
        true_indices = [
            i
            for i in range(
                len(jets)
            )
            if abs(
                jets[i][
                    0
                ].DeltaPhi(
                    jets[i][
                        1
                    ]
                )
            )
            >= Range[0]
            and abs(
                jets[i][
                    0
                ].DeltaPhi(
                    jets[i][
                        1
                    ]
                )
            )
            <= Range[1]
        ]
    elif bin_var == "m_jj":
        true_indices = [
            i
            for i in range(
                len(jets)
            )
            if np.sum(
                jets[i]
            ).M()
            >= Range[0]
            and np.sum(
                jets[i]
            ).M()
            <= Range[1]
        ]
    else:
        raise ValueError
    print(true_indices)
    binned_name = (
        bin_var
        + "_"
        + str(Range[0])
        + "_"
        + str(Range[1])
    )
    binned_events = {}
    for (
        key,
        val,
    ) in events.items():
        try:
            if (
                key
                == "EventAttribute"
            ):
                val = (
                    np.array(
                        val
                    )
                )
            binned_events[
                key
            ] = val[
                true_indices
            ]
        except (
            TypeError
        ) as e:
            print(e, key)
            if key in {
                "cut_flow"
            }:
                binned_events[
                    key
                ] = val
    return binned_events


def cut_counter(
    prev_cut_flow,
    current_cut_flow,
):
    if (
        len(
            prev_cut_flow.keys()
        )
        == 0
    ):
        return (
            current_cut_flow
        )
    else:
        for (
            key,
            val,
        ) in (
            current_cut_flow.items()
        ):
            if (
                type(val)
                != int
            ):
                print(
                    key,
                    " has value ",
                    type(
                        val
                    ),
                    " skipping...",
                )
                continue
            prev_cut_flow[
                key
            ] += val
        return prev_cut_flow


def cut_efficiency(
    cut_flow, verbose=False
):
    order = cut_flow["order"]
    tot = cut_flow["total"]
    efficiencies = {
        "total_efficiency": cut_flow[
            "passed"
        ]
        / tot
    }
    if verbose:
        print(
            "total_efficiency",
            efficiencies[
                "total_efficiency"
            ],
            "Total: ",
            cut_flow[
                "total"
            ],
            "Passed: ",
            cut_flow[
                "passed"
            ],
        )
    count = 0
    for key in order:
        # count+=cut_flow[key]
        efficiencies[key] = (
            cut_flow[key]
            / cut_flow[
                "total"
            ]
        )
        if verbose:
            print(
                key,
                " :",
                efficiencies[
                    key
                ],
            )
        # tot=tot-cut_flow[key]
    # if verbose: print ("Total rejected: ","\nRemaining: ",tot,"\nSum: ",count+tot)
    # assert count+tot==cut_flow["total"]
    return efficiencies


def dir_ext_count(
    ext,
    dir_path,
    prefix="",
    suffix="",
):
    path = []
    for item in os.listdir(
        dir_path
    ):
        if item.startswith(
            prefix
        ) and item.endswith(
            ext
        ):
            print(
                item, prefix
            )
            if (
                item[
                    -len(ext)
                    - len(
                        suffix
                    ) :
                ]
                == suffix
            ):
                path.append(
                    os.path.join(
                        dir_path,
                        item,
                    )
                )
    return path


def workaround_concatenate(
    to_return,
    to_append,
    return_type="array",
):
    return_array = []
    for item in to_return:
        return_array.append(
            item
        )
    for item in to_append:
        return_array.append(
            item
        )
    if (
        return_type
        != "array"
    ):
        return return_array
    else:
        return np.array(
            return_array
        )


def merge_flat_dict(
    append,
    temp,
    append_length=None,
    keys="all",
    exclude=[],
):
    """combine two dictionaries of same set of <keys> with <numpy.array> as values"""
    if (
        len(
            list(
                append.keys()
            )
        )
        == 0
    ):
        if keys == "all":
            return temp
        else:
            return_dict = {}
            for item in keys:
                return_dict[
                    item
                ] = temp[
                    item
                ]
    if keys == "all":
        keys = temp.keys()
    for item in append:
        if (
            item
            == "EventAttribute"
        ):
            print(
                "Found EventAttribute, concatenating as list..."
            )
            append[
                item
            ] = workaround_concatenate(
                append[item],
                temp[item][
                    :append_length
                ],
                return_type="list",
            )
            continue
        if (
            item not in keys
            or item
            in exclude
        ):
            continue
        if (
            type(
                append[item]
            )
            == np.ndarray
            and type(
                temp[item]
            )
            == np.ndarray
        ):
            try:
                append[
                    item
                ] = np.concatenate(
                    (
                        append[
                            item
                        ],
                        temp[
                            item
                        ][
                            :append_length
                        ],
                    ),
                    axis=0,
                )
            except (
                ValueError
            ) as e:
                print(
                    e,
                    "trying workaround method",
                )
                append[
                    item
                ] = workaround_concatenate(
                    append[
                        item
                    ],
                    temp[
                        item
                    ][
                        :append_length
                    ],
                )
        elif (
            type(
                append[item]
            )
            == list
            and type(
                temp[item]
            )
            == list
        ):
            if (
                "debug"
                in sys.argv
            ):
                print(
                    "list",
                    item,
                    len(
                        append[
                            item
                        ]
                    ),
                    len(
                        temp[
                            item
                        ]
                    ),
                    append[
                        item
                    ][
                        0
                    ].shape,
                    append[
                        item
                    ][
                        1
                    ].shape,
                    temp[
                        item
                    ][
                        0
                    ].shape,
                    temp[
                        item
                    ][
                        1
                    ].shape,
                )
            assert len(
                append[item]
            ) == len(
                temp[item]
            )
            for i in range(
                len(
                    append[
                        item
                    ]
                )
            ):
                append[item][
                    i
                ] = np.concatenate(
                    (
                        append[
                            item
                        ][
                            i
                        ],
                        temp[
                            item
                        ][i][
                            :append_length
                        ],
                    ),
                    axis=0,
                )
            if (
                "debug"
                in sys.argv
            ):
                print(
                    "list",
                    item,
                    len(
                        append[
                            item
                        ]
                    ),
                    len(
                        temp[
                            item
                        ]
                    ),
                    append[
                        item
                    ][
                        0
                    ].shape,
                    append[
                        item
                    ][
                        1
                    ].shape,
                    temp[
                        item
                    ][
                        0
                    ].shape,
                    temp[
                        item
                    ][
                        1
                    ].shape,
                )
            # sys.exit()
    return append


def print_events(
    events, name=None
):
    """Function for printing nested dictionary with atmost 3 levels, with final value being a numpy.ndarry, prints the shape of the array"""
    if name:
        print(name)
    for channel in events:
        if hasattr(
            events[channel],
            "shape",
        ) or hasattr(
            events[channel],
            "__len__",
        ):
            if hasattr(
                events[
                    channel
                ],
                "shape",
            ):
                print(
                    "    Final State:",
                    channel,
                    events[
                        channel
                    ].shape,
                    f" dtype: {type(events[channel])}",
                )
            elif (
                channel
                == "EventAttribute"
            ):
                print(
                    "    Final State: ",
                    channel,
                    np.array(
                        events[
                            channel
                        ]
                    ).shape,
                    " dtype: EventAttribute",
                )
            else:
                # try: print ("    Final State:", channel,[item.shape for item in events[channel]],f' dtype: {type(events[channel])}')
                # except AttributeError:
                print(
                    "    Final State:",
                    channel,
                    len(
                        events[
                            channel
                        ]
                    ),
                    f" dtype: {type(events[channel])}",
                )
            continue
        print(
            "Specs: ",
            channel,
            "\n     Content: ",
            events[channel],
        )
    return


def check_file(
    name,
    event_folder,
    tag="",
    full_name=False,
    suffix=False,
    run_tag="None",
    target_file=None,
):
    """at any <event_folder>(madgraph event_folder) go to each <run_dir> to get relative
    path from event_folder to any file with any <extension> . Choose different madgraph tags with <tag>
    """
    path = []
    pwd = os.getcwd()
    if (
        pwd
        != os.path.abspath(
            event_folder
        )
    ):
        os.chdir(
            event_folder
        )
    # print (event_folder,os.listdir(event_folder))
    for item in os.listdir(
        event_folder
    ):
        # print (os.getcwd()+"/"+item,os.path.isdir(os.getcwd()+"/"+item))
        if os.path.isdir(
            item
        ):
            files = (
                os.listdir(
                    item
                )
            )
            if (
                run_tag
                != "None"
            ):
                if (
                    run_tag
                    not in {
                        item[
                            : len(
                                run_tag
                            )
                        ]
                        for item in files
                    }
                ):
                    continue
            os.chdir(item)
            assert os.access(
                ".", os.W_OK
            ), os.getcwd()
            if (
                target_file
                is not None
            ):
                if (
                    target_file
                    in files
                ):
                    print(
                        target_file,
                        " found in ",
                        os.getcwd(),
                        " skipping ",
                    )
                    os.chdir(
                        ".."
                    )
                    continue
            for (
                filename
            ) in files:
                if full_name:
                    if (
                        filename
                        == name
                    ):
                        path.append(
                            os.path.join(
                                event_folder,
                                item,
                                filename,
                            )
                        )
                    continue
                if (
                    not suffix
                ):
                    if (
                        filename[
                            -len(
                                name
                            ) :
                        ]
                        == name
                        and filename[
                            : len(
                                tag
                            )
                        ]
                        == tag
                    ):
                        path.append(
                            os.path.join(
                                event_folder,
                                item,
                                filename,
                            )
                        )
                    continue
                if (
                    filename[
                        -len(
                            name
                        ) :
                    ]
                    == name
                    and filename[
                        len(
                            tag
                        ) :
                    ]
                    == tag
                ):
                    path.append(
                        os.path.join(
                            event_folder,
                            item,
                            filename,
                        )
                    )
            os.chdir("..")
    os.chdir(pwd)
    path.sort()
    return path


def check_dir(path):
    """check if <path> to dir exists or not. If it doesn't, create the <dir> returns the absolute path to the created dir"""
    pwd = os.getcwd()
    try:
        os.chdir(path)
    except OSError:
        os.mkdir(path)
        os.chdir(path)
    path = os.getcwd()
    os.chdir(pwd)
    return path


def arg_split(
    args,
    num_cores,
    ignore_keys=["cut_flow"],
    verbose=False,
):
    """to fix: not splitting args with len(array)==num_cores correctly into iterables of single length"""
    print(
        "Arge type: ",
        type(args),
    )
    if (
        type(args)
        == np.ndarray
        or type(args) == list
    ):
        step = int(
            len(args)
            / num_cores
        )
        arg = []
        for i in range(
            0,
            len(args),
            step,
        ):
            try:
                arg.append(
                    args[
                        i : i
                        + step
                    ]
                )
            except (
                IndexError
            ):
                arg.append(
                    args[i:]
                )
    elif type(args) == dict:
        arg = [
            dict()
            for i in range(
                num_cores
            )
        ]
        if verbose:
            print(
                "Splitting argument into: ",
                len(arg),
            )
            print_events(
                args,
                name="Original:",
            )
        start_inds = []
        for key in args:
            if (
                key
                in ignore_keys
            ):
                print(
                    "Ignoring key: ",
                    key,
                )
                continue
            if (
                type(
                    args[key]
                )
                == np.ndarray
            ):
                step = (
                    int(
                        len(
                            args[
                                key
                            ]
                        )
                        / num_cores
                    )
                    + 1
                )
            else:
                step = int(
                    len(
                        args[
                            key
                        ]
                    )
                    / num_cores
                )
            count = 0
            # print (key,step,np.array(args[key]).shape)
            for key in args:
                if (
                    key
                    in ignore_keys
                ):
                    print(
                        "Ignoring key: ",
                        key,
                    )
                    continue
                step = (
                    int(
                        len(
                            args[
                                key
                            ]
                        )
                        / num_cores
                    )
                    + 1
                )
                count = 0
                for (
                    i
                ) in range(
                    0,
                    len(
                        args[
                            key
                        ]
                    ),
                    step,
                ):
                    if (
                        i
                        not in start_inds
                    ):
                        start_inds.append(
                            i
                        )
                    arg[
                        count
                    ][
                        key
                    ] = args[
                        key
                    ][
                        i : i
                        + step
                    ]
                    count += (
                        1
                    )

        if verbose:
            core_ind = 0
            print(
                "Splitted dictionaries with: Arg_<start_ind>_<core_ind>"
            )
            for (
                start,
                item,
            ) in zip(
                start_inds,
                arg,
            ):
                print_events(
                    item,
                    name="Arg_"
                    + str(
                        start
                    )
                    + "_"
                    + str(
                        core_ind
                    ),
                )
            core_ind += 1
    else:
        raise TypeError(
            "No algorithm for splitting arguments of type: "
            + type(args)
        )
    return arg


def init_lock(l):
    global lock
    lock = l


def pool_splitter(
    function,
    args,
    num_cores=multiprocessing.cpu_count(),
    exclude=[],
    ignore_keys=["cut_flow"],
    add_keys=[],
    with_lock=False,
    verbose=False,
):
    """utility function for multiprocessing any function with single argument of either numpy.ndarray or flat dict with numpy.ndarray values"""
    add_dict = {}
    for item in (
        ignore_keys
        + add_keys
    ):
        if item in args:
            add_dict[
                item
            ] = args.pop(
                item
            )
    arg = arg_split(
        args,
        num_cores,
        ignore_keys=ignore_keys,
        verbose=verbose,
    )
    arg = [
        item
        for item in arg
        if item
    ]
    for (
        key,
        val,
    ) in add_dict.items():
        if key in add_keys:
            for item in arg:
                item[
                    key
                ] = val
    if len(arg) < num_cores:
        print(
            "Not enough arguments to split in :",
            num_cores,
            " cores\nReducing number of cores",
        )
        num_cores = len(arg)
    if with_lock:
        print(
            "Initialising lock..."
        )
        l = (
            multiprocessing.Lock()
        )
        p = multiprocessing.Pool(
            processes=num_cores,
            initializer=init_lock,
            initargs=(l,),
        )
    else:
        p = multiprocessing.Pool(
            processes=num_cores
        )
    if num_cores > 1:
        print(
            "Splitting "
            + function.__name__
            + " on "
            + str(num_cores)
            + " cores..."
        )
    try:
        data = p.map(
            function, arg
        )
    except (
        KeyboardInterrupt
    ) as ki:
        p.close()
        raise ki
    finally:
        p.close()
    print("Done!")
    if (
        type(data[0])
        == np.ndarray
    ):
        return (
            np.concatenate(
                data, axis=0
            )
        )
    elif (
        type(data[0]) == dict
    ):
        # if len(data)>16:
        #    print ("Splitting concatenation on 8 cores...")
        #    p=multiprocessing.Pool(processes=8)
        #    new_args=arg_split(data,8)
        #    data=p.map(combine_dict,new_args)
        #    p.close()
        data = combine_dict(
            data,
            exclude=exclude,
        )
        data.update(add_dict)
        return data
    else:
        return data


def seperate_classes(
    data, class_names
):
    print(
        "Seperating classes..."
    )
    X, Y = (
        data["X"],
        data["Y"],
    )
    (
        class_0,
        class_1,
    ) = np.nonzero(
        Y[:, 0]
    ), np.nonzero(
        Y[:, 1]
    )
    if "debug" in sys.argv:
        print(
            type(class_0),
            len(class_0[0]),
            class_0[0][:2],
            Y[
                class_0[0][
                    :2
                ]
            ],
        )
    X0, Y0 = (
        X[class_0],
        Y[class_0],
    )
    X1, Y1 = (
        X[class_1],
        Y[class_1],
    )
    # if not class_names: self.class_names=("class_0","class_1")
    return {
        class_names[0]: {
            "X": X0,
            "Y": Y0,
        },
        class_names[1]: {
            "X": X1,
            "Y": Y1,
        },
    }


def concatenate_list(
    data,
    verbose=False,
    **kwargs,
):
    count = 0
    if verbose:
        for item in data:
            print(
                "-------------------------------------------------"
            )
            for (
                particle
            ) in item:
                try:
                    print(
                        particle,
                        item[
                            particle
                        ].shape,
                    )
                except AttributeError:
                    print(
                        particle,
                        len(
                            item[
                                particle
                            ]
                        ),
                    )
    return_dict = {}
    for item in data:
        for particle in item:
            if (
                particle
                not in return_dict
            ):
                return_dict[
                    particle
                ] = []
            for (
                event
            ) in list(
                item[
                    particle
                ]
            ):
                return_dict[
                    particle
                ].append(
                    event
                )
    for item in return_dict:
        return_dict[
            item
        ] = np.array(
            return_dict[item]
        )
    if "cut_flow" in kwargs:
        return_dict[
            "cut_flow"
        ] = kwargs.get(
            "cut_flow"
        )
    return return_dict


def combine_dict(
    data, exclude=[]
):
    print(
        "Combining dictionaries..."
    )
    # sys.exit()
    return_dict = dict()
    cont, count = False, 0
    if "cut_flow" in data[0]:
        sum_dicts = [
            data[i].pop(
                "cut_flow"
            )
            for i in range(
                len(data)
            )
        ]
        cut_flow = {}
        for (
            item
        ) in sum_dicts:
            cut_flow = (
                cut_counter(
                    cut_flow,
                    item,
                )
            )
        add = True
    else:
        add = False
    for i in range(
        len(data)
    ):
        if (
            len(
                data[
                    i
                ].keys()
            )
            == 0
        ):
            continue
        for key in data[i]:
            if (
                key
                in exclude
            ):
                continue
            if count == 0:
                return_dict[
                    key
                ] = data[i][
                    key
                ]
            else:
                if (
                    type(
                        data[
                            i
                        ][
                            key
                        ]
                    )
                    == list
                ):
                    return_dict[
                        key
                    ] = (
                        return_dict[
                            key
                        ]
                        + data[
                            i
                        ][
                            key
                        ]
                    )
                else:
                    try:
                        return_dict[
                            key
                        ] = np.concatenate(
                            (
                                return_dict[
                                    key
                                ],
                                data[
                                    i
                                ][
                                    key
                                ],
                            ),
                            axis=0,
                        )
                    except ValueError as E:
                        print(
                            E,
                            "\nCould not concatenate as numpy array, appending as lists...",
                        )
                        if add:
                            kwargs = {
                                "cut_flow": cut_flow
                            }
                        return concatenate_list(
                            data,
                            **kwargs,
                        )
        count += 1
    for key in exclude:
        return_dict[
            key
        ] = data[0][key]
    if add:
        return_dict[
            "cut_flow"
        ] = cut_flow
    return return_dict

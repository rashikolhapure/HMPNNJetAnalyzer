import numpy as np
from itertools import (
    combinations,
    permutations,
)


class Legend:
    LhcIndex = {
        0: "Eta",
        1: "Phi",
        2: "pt",
        3: "E",
    }
    Metric = np.array([1, -1, -1, -1])


def MinkowskiDot(a, b):
    """
    Compute the Minkowski dot product between two 4-vectors.

    Parameters:
    ----------
    a : numpy.ndarray
        The first 4-vector.
    b : numpy.ndarray
        The second 4-vector.

    Returns:
    -------
    numpy.ndarray
        The Minkowski dot product between 'a' and 'b'.

    Raises:
    ------
    AssertionError
        If the shapes of 'a' and 'b' do not match or if the shape[-1] is
        not 4 (no Lorentz axis).

    Notes:
    -----
    This function computes the Minkowski dot product between two 4-vectors,
    which are typically used in special relativity calculations.
    The dot product is calculated as the difference of the first
    components multiplied by each other minus the sum of the
    products of the remaining components.
    """
    assert a.shape == b.shape and a.shape[-1] == 4, "No Lorentz Axis"
    InitShape = a.shape
    ReturnShape = tuple(InitShape[i] for i in range(len(InitShape) - 1))
    print(
        InitShape,
        ReturnShape,
    )
    a, b = a.reshape(-1, 4), b.reshape(-1, 4)
    ReturnArray = np.zeros(a.shape[0])
    for i in range(len(ReturnArray)):
        ReturnArray[i] = a[i, 0] * b[i, 0] - np.sum(a[i, 1:] * b[i, 1:])
    print(
        ReturnArray,
        "\n",
        ReturnArray.reshape(ReturnShape),
    )
    return ReturnArray.reshape(ReturnShape)


def ConvertToLhc(Array):
    """
    Convert an array of 4-vectors to the Lorentz-Heaviside coordinate system.

    Parameters:
    ----------
    Array : numpy.ndarray
        The array of 4-vectors to be converted.

    Returns:
    -------
    numpy.ndarray
        The converted array in the Lorentz-Heaviside coordinate system.

    Raises:
    ------
    AssertionError
        If the shape[-1] of 'Array' is not 4 (no Lorentz axis).

    Notes:
    -----
    This function converts an array of 4-vectors to the Lorentz-Heaviside
    coordinate system, which is commonly used in high-energy physics.
    The Lorentz-Heaviside coordinates are defined in terms of rapidity
    (eta), azimuthal angle (phi), transverse momentum (pt), and energy (E).
    """
    assert Array.shape[-1] == 4, "No Lorentz Axis"
    InitShape = Array.shape
    Array = Array.reshape(-1, 4)
    # print (Array)
    # Array=Array.reshape(InitShape)
    # print (Array)
    # sys.exit()
    ReturnArray = np.zeros((Array.shape))
    for i in range(len(Array)):
        ReturnArray[i] = np.array(
            [
                -np.log(
                    np.tan(
                        np.arccos(
                            Array[
                                i,
                                3,
                            ]
                            / np.sqrt(
                                Array[
                                    i,
                                    1,
                                ]
                                ** 2
                                + Array[
                                    i,
                                    2,
                                ]
                                ** 2
                                + Array[
                                    i,
                                    3,
                                ]
                                ** 2
                            )
                        )
                        / 2.0
                    )
                ),
                np.arctan(Array[i, 2] / Array[i, 1]),
                np.sqrt(Array[i, 1] ** 2 + Array[i, 2] ** 2),
                Array[i, 0],
            ]
        )
    # print ("CONVERT",ReturnArray.reshape(InitShape),"\n")
    return ReturnArray.reshape(InitShape)


def Boost(particle, direction, eta):
    """
    Apply a Lorentz boost to a particle's 4-momentum.

    Parameters:
    ----------
    particle : numpy.ndarray
        The 4-momentum of the particle, typically in the form [E, px, py, pz].
    direction : numpy.ndarray
        The direction vector along which the boost is applied, typically
        a 3D vector.
    eta : float
        The rapidity parameter for the boost.

    Returns:
    -------
    numpy.ndarray
        The boosted 4-momentum of the particle.

    Raises:
    ------
    AssertionError
        If the magnitude of the 'direction' vector is not approximately 1.
        If the lengths of 'particle' and 'direction' vectors are not valid
        (4 and 3, respectively).

    Notes:
    -----
    This function applies a Lorentz boost to the given particle's 4-momentum
    in the specified direction and rapidity. The 'direction' vector should be
    normalized (magnitude approximately 1) before calling this function.
    """
    assert abs(np.sum(direction**2) - 1.0) < 1e-12
    (
        particle,
        direction,
    ) = np.array(
        particle
    ), np.array(direction)
    assert len(particle) == 4 and len(direction) == 3
    E, p = particle[0], Euclid3Norm(particle[1:] * direction)
    # print (particle,E,p)
    return np.array(
        [
            E * np.cosh(eta) + p * np.sinh(eta),
            E * np.sinh(eta) + p * np.cosh(eta),
        ]
    )


def SumCombinations(
    FourVectors,
    Map=None,
    comb=2,
):
    """
    Calculate the sum of combinations of four-vectors.

    Parameters:
    ----------
    FourVectors : numpy.ndarray
        An array containing four-vectors.
    Map : iterable, optional
        A custom mapping of combinations (default is None).
    comb : int, optional
        The number of four-vectors to combine in each combination
        (default is 2).

    Returns:
    -------
    numpy.ndarray or tuple
        If 'Map' is not provided, returns an array containing the sums of
        combination of all possible pairs of four-vectors. Otherwise it will
        return an array containing the sums of combinations of four-vectors
        If 'Map' is provided, returns a tuple containing the array and the
        original combination mapping.

    Raises:
    ------
    AssertionError
        If 'FourVectors' is not a 2D array with shape[1] equal to 4
        (invalid four-vectors).

    Notes:
    -----
    This function calculates the sum of combinations of four-vectors
    from 'FourVectors'. 'comb' specifies the number of four-vectors
    to combine in each combination. 'Map' can be provided to specify
    a custom mapping of combinations, or it will be generated if not
    provided.
    """
    assert (
        len(FourVectors.shape) == 2 and FourVectors.shape[1] == 4
    ), "Invalid argument as FourVectors"
    if Map is None:
        Map = list(
            combinations(
                np.arange(len(FourVectors)),
                comb,
            )
        )
        ReturnMap = True
    else:
        Map = list(Map)
        ReturnMap = False
    ReturnArray, count = (
        np.zeros(
            (len(Map), 4),
            dtype="float64",
        ),
        0,
    )
    for item in Map:
        ReturnArray[count] = np.sum(
            np.take(
                FourVectors,
                item,
                axis=0,
            ),
            axis=0,
        )
        count += 1
    if ReturnMap:
        return (
            ReturnArray,
            tuple(Map),
        )
    else:
        return ReturnArray


def UnequalSet(*args):
    for i in range(len(args) - 1):
        assert len(list(args[i])) == len(list(args[i + 1])) and isinstance(
            args[i], type(args[i + 1])
        )
        for item in list(args[i]):
            assert args[i].count(item) == 1
            if item in list(args[i + 1]):
                return False
    else:
        return True


def MapDict(Map):
    """
    Create a dictionary of unique combinations from a given mapping.

    Parameters:
    ----------
    Map : iterable
        The mapping of combinations.

    Returns:
    -------
    dict
        A dictionary containing unique combinations from the input mapping.

    Notes:
    -----
    This function creates a dictionary containing unique combinations from
    the input 'Map'. Each unique combination is stored as a list in
    the dictionary.
    """
    ReturnDict, count = (
        dict(),
        0,
    )
    for i in range(len(Map)):
        for j in range(i + 1, len(Map)):
            if UnequalSet(
                Map[i],
                Map[j],
            ):
                ReturnDict["Map_" + str(count)] = [
                    Map[i],
                    Map[j],
                ]
                count += 1
    return ReturnDict


def GetMass(particle):
    """
    Calculate the invariant mass of a particle or an array of particles.

    Parameters:
    ----------
    particle : numpy.ndarray
        The 4-momentum of the particle or an array of particles.

    Returns:
    -------
    float or numpy.ndarray
        The invariant mass of the particle(s).

    Raises:
    ------
    AssertionError
        If the shape[-1] of 'particle' is not 4.

    Notes:
    -----
    This function calculates the invariant mass of a single particle or an
    array of particles. The input 'particle' is expected to be a 4-momentum,
    where the first element is energy (E) and the remaining elements are the
    spatial components (px, py, pz). The invariant mass is computed
    using the relativistic formula.
    """
    assert particle.shape[-1] == 4
    if len(particle.shape) == 1:
        return particle[0] * np.sqrt(
            1 - np.sum(particle[1:] ** 2) / particle[0] ** 2
        )
    else:
        init_shape = list(particle.shape)
        # print (particle)
        particle = particle.reshape(-1, 4)
        return_array = np.zeros(particle.shape[0])
        count = 0
        for item in particle:
            return_array[count] = item[0] * np.sqrt(
                1 - np.sum(item[1:] ** 2) / item[0] ** 2
            )
            count += 1
        return_array = return_array.reshape(tuple(init_shape[:-1]))
        return return_array


def Get3Direction(
    FourVector,
):
    """
    Calculate the 3-direction vector from a 4-vector.

    Parameters:
    ----------
    FourVector : numpy.ndarray
        The 4-vector from which the 3-direction vector is calculated.

    Returns:
    -------
    numpy.ndarray
        The 3-direction vector.

    Raises:
    ------
    AssertionError
        If the length of 'FourVector' is not 4.
        If the calculated 3-direction vector's magnitude is not approximately
        1.

    Notes:
    -----
    This function calculates the 3-direction vector from a 4-vector.
    The input 'FourVector' is expected to be a 4-momentum, where the
    first element is energy (E) and the remaining elements are the spatial
    components (px, py, pz). The 3-direction vector is obtained by normalizing
    the spatial components.
    """
    assert len(FourVector) == 4
    Dir = FourVector[1:] / Euclid3Norm(FourVector)
    assert abs(Euclid3Norm(Dir) - 1) < 1e-12
    return Dir


def GetEta(FourVector):
    """
    Calculate the pseudorapidity (eta) of a 4-vector.

    Parameters:
    ----------
    FourVector : numpy.ndarray
        The 4-vector for which the pseudorapidity (eta) is calculated.

    Returns:
    -------
    float
        The pseudorapidity (eta) of the 4-vector.

    Raises:
    ------
    AssertionError
        If the length of 'FourVector' is not 4.

    Notes:
    -----
    This function calculates the pseudorapidity (eta) of a 4-vector.
    The input 'FourVector' is expected to be a 4-momentum, where the
    first element is energy (E) and the remaining elements are the spatial
    components (px, py, pz). The pseudorapidity is computed using the
    arctanh function.
    """
    assert len(FourVector) == 4
    return np.arctanh(Euclid3Norm(FourVector) / FourVector[0])


def Euclid3Norm(FourVector):
    """
    Calculate the Euclidean 3-norm of a 4-vector.

    Parameters:
    ----------
    FourVector : numpy.ndarray
        The 4-vector for which the Euclidean 3-norm is calculated.

    Returns:
    -------
    float or numpy.ndarray
        The Euclidean 3-norm of the 4-vector or an array of 3-norms.

    Notes:
    -----
    This function calculates the Euclidean 3-norm ofa 4-vector. If'FourVector'
    is a 3-component vector, it is treated as a spatial vector, and the time
    component is added as 0.0 before computing the 3-norm. If 'FourVector' is a
    multi-component array of 4-vectors, the 3-norm is computed for each
    4-vector along the last axis.
    """
    if len(FourVector.shape) == 1 and len(FourVector) == 3:
        FourVector = np.concatenate(
            (
                [0.0],
                FourVector,
            ),
            axis=0,
        )
        return np.sqrt(
            np.sum(
                FourVector[:3] ** 2,
                axis=0,
            )
        )
    return_array = np.zeros(
        (len(FourVector)),
        dtype="float64",
    )
    return_array = np.sum(
        FourVector[:3,] ** 2,
        axis=0,
    )
    print(return_array)
    return return_array

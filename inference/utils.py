import numpy as np


def calculate_roc(
    data,
    signal="signal",
    background="background",
):
    """
    Calculate ROC (Receiver Operating Characteristic) values based on
    signal and background data.

    Args:
        data (dict): A dictionary containing signal and background data.
        signal (str): Key for the signal data in the data dictionary.
        background (str): Key for the background data in the data dictionary.

    Returns:
        tuple: A tuple containing two lists - signal efficiency and
            background rejection.
    """
    print("calculating roc values...")
    sg = data[signal][:, 0]
    bg = data[background][:, 0]
    sig_eff, bg_rej = [], []
    for threshold in np.linspace(0.001, 1, 100):
        sig_eff.append(np.count_nonzero(np.where(sg > threshold)) / len(sg))
        bg_rej.append(
            len(bg) / (np.count_nonzero(np.where(bg > threshold)) + 10e-12)
        )
    return sig_eff, bg_rej


def root_mean_squared_error(true, pred):
    """
    Calculate the root mean squared error (RMSE) between true and
    predicted values.

    Args:
        true (numpy.ndarray): The true values.
        pred (numpy.ndarray): The predicted values.

    Returns:
        numpy.ndarray: An array of RMSE values for each pair of true
            and predicted values.
    """
    print(
        true.shape,
        pred.shape,
    )
    assert true.shape == pred.shape
    ReturnArray = np.zeros(len(true))
    count = 0
    for x, y in zip(true, pred):
        ReturnArray[count] = np.sqrt(np.sum((x - y) ** 2) / np.sum(x**2))
        count += 1
    return ReturnArray

import tensorflow.keras as keras
import sys
import matplotlib.pyplot as plt
import numpy as np
import __main__
from itertools import combinations


def predict_enc(
    enc, data_dict, subplots=False, type="scatter", save_path="./plots", **kwargs
):
    """Predicts the encoded values for each data item in data_dict using the encoder enc, and generates scatter or binned plots for the pairwise combinations of these encoded values.

    Parameters:
    enc (keras.Model): The encoder model used to encode the data items.
    data_dict (dict): A dictionary containing the data items to be encoded and plotted. The keys are the names of the items and the values are numpy arrays.
    subplots (bool, optional): If True, the plots will be generated in subplots. Defaults to False.
    type (str, optional): The type of plot to generate. "scatter" for scatter plots, "binned" for binned plots. Defaults to "scatter".
    save_path (str, optional): The path to save the generated plots. Defaults to "./plots".
    **kwargs: Additional arguments to be passed to the scatter or binned plot functions.

    Returns:
    None
    """
    return_array = []
    class_count = 0
    names = [item for item in data_dict]
    # names.reverse()
    print(names)
    new_dict = {item: [] for item in data_dict}
    for item in data_dict:
        lat_dim = data_dict[item].shape[-1]
        temp = np.swapaxes(enc.predict(data_dict[item], verbose=1), 0, 1)
        print(item, temp.shape, np.mean(temp, axis=1), np.std(temp, axis=1))
        # sys.exit()
        if type == "binned":
            new_dict[item] = temp
            num_plots = temp.shape[0]
            continue
        for combs in combinations(temp, 2):
            print(combs[0].shape, new_dict[item])
            # sys.exit()
            new_dict[item].append([combs[0], combs[1]])
        print(len(list(combinations([i for i in range(len(temp))], 2))))
        num_plots = len(new_dict[item])
    print(num_plots)
    for item in new_dict:
        array = new_dict[item]
        print(item, np.mean(array, axis=1), np.std(array, axis=1))
    if type == "scatter":
        scatter(new_dict, num_plots, subplots, save_path, order=names, **kwargs)
    elif type == "binned":
        binner(new_dict, num_plots, subplots, save_path, **kwargs)
    return
    # sys.exit()


def scatter(new_dict, num_plots, subplots, save_path, order=None, suffix=""):
    """
    The function scatter generates scatter plots of latent space coordinates for each pair of inputs in new_dict. It takes the following parameters:

    parameters:

    new_dict: a dictionary containing the latent space coordinates for each input pair
    num_plots: an integer representing the number of plots to generate
    subplots: a boolean indicating whether to generate subplots
    save_path: a string representing the directory where the plots will be saved
    order: a list containing the order in which to plot the inputs
    suffix: a string to add to the end of the filename

    The function generates scatter plots of latent space coordinates, either in individual plots or in subplots depending on the subplots parameter. If subplots are used, a single plot is generated with all subplots, otherwise, individual plots are generated for each scatter plot. The plots are saved in the directory specified by save_path. If order is specified, the inputs are plotted in the specified order, otherwise, the inputs are plotted in the order they appear in the new_dict dictionary. The suffix parameter can be used to add a string to the end of the filename.
    """
    rows = int(np.sqrt(num_plots)) + 1
    for i in range(num_plots):
        if subplots:
            plt.subplot(rows, rows, i + 1)
        for item in order:
            plt.scatter(new_dict[item][i][0], new_dict[item][i][1], label=item)
        plt.legend(loc="best")
        if not subplots:
            try:
                plt.savefig(
                    save_path + "/latent_scatter_" + str(i) + "_" + suffix + ".png",
                    format="png",
                    dpi=1000,
                )
            except:
                pass
            plt.show(block=False)
            plt.close()
    if subplots:
        try:
            plt.savefig(
                save_path + "/latent_" + suffix + ".png", format="png", dpi=1000
            )
        except:
            pass
        plt.show(block=False)
        plt.close()
    return


def binner(data, num_plots, subplots, save_path, bins=50, suffix=""):
    """
    The function binner generates binned plots of latent space coordinates for each input in data. It takes the following parameters:

    parameters:

    data: a dictionary containing the latent space coordinates for each input
    num_plots: an integer representing the number of plots to generate
    subplots: a boolean indicating whether to generate subplots
    save_path: a string representing the directory where the plots will be saved
    bins: an integer representing the number of bins to use in the binned plots
    suffix: a string to add to the end of the filename

    Returns:
    None

    The function generates binned plots of latent space coordinates, either in individual plots or in subplots depending on the subplots parameter. If subplots are used, a single plot is generated with all subplots, otherwise, individual plots are generated for each binned plot. The plots are saved in the directory specified by save_path. The suffix parameter can be used to add a string to the end of the filename.
    """
    rows = int(np.sqrt(num_plots)) + 1
    for i in range(num_plots):
        if subplots:
            plt.subplot(rows, rows, i + 1)
        for item in data:
            plt.hist(data[item][i], bins=bins, histtype="step", label=item)
        plt.legend(loc="best")
        if not subplots:
            try:
                plt.savefig(
                    save_path + "/latent_" + str(i) + "_" + suffix + ".png",
                    format="png",
                    dpi=1000,
                )
            except:
                pass
            plt.show(block=False)
            plt.close()
    if subplots:
        try:
            plt.savefig(
                save_path + "/latent_" + suffix + ".png", format="png", dpi=1000
            )
        except:
            pass
        plt.show(block=False)
        plt.close()
    return


def transfer_weights(trained, model, verbose=False):
    """
    The function transfer_weights takes in two Keras models, trained and model, and transfers the weights of the layers from the trained model to the corresponding layers in the model based on their order.

    Parameters:

    trained (keras.models.Model): the pre-trained model from which weights will be transferred.
    model (keras.models.Model): the model to which weights will be transferred.
    verbose (bool): if True, prints the weights before and after they are set. Default is False.

    Returns:

    model (keras.models.Model): the model with updated weights.

    """
    enc_layers = model.layers
    count = 0
    for layer in trained.layers:
        temp = layer.get_weights()
        if verbose:
            print("temp", temp)
        model.layers[count].set_weights(temp)
        if verbose:
            print("set", model.layers[count].get_weights())
        count += 1
        if count == len(enc_layers):
            break
    return model


if __name__ == "__main__":
    model = keras.models.load_model("./model.h")
    encoder = keras.models.load_model("./enc.h")
    data_sg = np.load("./data/vvz.npy")[:180000]
    data_bg = np.load("./data/zjj.npy")[:180000]
    data_sg, data_bg = np.expand_dims(data_sg, -1), np.expand_dims(data_bg, -1)
    data = {"V jets": data_sg, "QCD jets": data_bg}
    for item in data:
        print(item, np.mean(data[item]), np.std(data[item]))
    # sys.exit()
    encoder = transfer_weights(model, encoder, verbose=False)
    predict_enc(encoder, data, subplots=False, type="binned", save_path="./plots")

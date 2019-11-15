# plot_community_dist.py

import pickle
import matplotlib.pyplot as plt

def community_dist_bar(colors, title=None, logscale=True, barh=False, show=True,
                        save=False, **kwargs):
    """
    Creates a community size distribution bar plot.

    Parameters:

        colors: dict/list of coloring data, or path (str) to pickled data.
            Accepted formats:
                dict:   {comm_num: array(nodes_in_comm)}
                        {comm_size: num_comms}

        ######## IGNORE FOR NOW. NOT IMPLEMENTED. PROBABLY NOT NEEDED. ########
                list:   [num_nodes_in_comm]

            If a list is passed, the value at index k must be the number of nodes
                in communities of size k+1; community sizes with no nodes in them
                should be filled with zeros.
         #######################################################################

        title (str): If str is passed, sets the plot title to 'title'

        logscale (bool): Default True. If False, bars are linearly scaled.

        barh (bool): Default False. If True, creates a horizontal bar graph.

        show (bool). Default True. If False, plot is generated but not displayed.
            Useful when generating and saving plots in batch.

        save (bool/str): Default False. If str, should be a path to location to
            save the plot.

        **kwargs: other keyword arguments to pass to plt.bar

    Returns: None
    """
    # if show is False, use gui-less backend
    if not show:
        plt.switch_backend('agg')

    # if colors is path data, load it
    if type(colors) is str:
        # load file
        with open(colors, 'rb') as f:
            colors = pickle.load(f)

    # colors should now be a dict or list
    if type(colors) is dict:
        # type of dict values should be int or array (or tuple)
        dict_type = type(list(colors.values())[0])
        if dict_type is int:
            # if dict_type is int, it maps comm_size --> num_comms
            # comm_size is bar positions, num_comms is bar heights
            comm_size, num_comms = colors.keys(), colors.values()
        else:
            # if dict_type is array or tuple, we have to extract comm_sizes
            # get largest community size (+1 for range indexing)
            max_comm_size = max([len(comm) for comm in list(colors.values())])+1
            # convert dict to {comm_size: num_comms}
            colors = {i: sum([1 for k in list(colors.keys()) if len(colors[k])==i])
                        for i in range(1, max_comm_size)}
            # remove comm_sizes with num_comms=0 to reduce plotting spatial complexity
            colors = {size: num for (size, num) in colors.items() if num!=0}
            # comm_size is bar positions, num_comms is bar heights
            comm_size, num_comms = colors.keys(), colors.values()

    elif type(colors) is list:
        raise NotImplementedError('list support is not implemented')

    # create plot
    if barh:
        raise NotImplementedError('barh plots not implemented')
    else:
        plt.bar(comm_size, num_comms, log=logscale, **kwargs)

    if title:
        plt.title(title)

    # save plot if specified
    if save:
        plt.savefig(save)

    # display spot
    if show:
        plt.show()

import matplotlib.pyplot as plt



def hsne(adata, channels_to_plot=None, scale_num=-1, subplot_grid_dim=(1, 1)):
    '''

    Parameters
    ----------
    adata
       anndata object
    channels_to_plot
       optional list of channel names, that can be plotted. If Null, than every channel will be plotted
    scale_num
       scale number that will be plotted
    subplot_grid_dim
       dimensions of the subplot grid
    -------
    Usage:
        Given an anndata object "adata" containing a HSNE embedding
        (calculate embedding first with the hsne(...) method in tools)
        => pl.hsne(adata)
    '''
    try:
        adata.uns['hsne_scales']
    except KeyError as e:
        raise Exception("scales have to be calculated first with tools.hsne")
    scales = adata.uns['hsne_scales']

    settings = adata.uns['hsne_settings']
    all_channel_names = list(adata.var_names.values)
    all_calculated_channel_names = adata.var_names.values  # available channels
    if channels_to_plot is None:
        channels_to_plot = all_calculated_channel_names

    for channel_name in channels_to_plot:
        if channel_name not in all_calculated_channel_names:
            raise Exception("Channel with name %s not found in \n%s"%(channel_name, all_calculated_channel_names))

    channels_to_plot_ind = [all_channel_names.index(name) for name in channels_to_plot] #all_calculated_channel_names]

    # dynamically add rows and cols for subplots
    num_channels = len(channels_to_plot)
    r, c = subplot_grid_dim
    if r == -1 or c == -1 or r * c < num_channels:
        while r * c < num_channels:
            if r > c:
                c += 1
            else:
                r += 1
    # create subplots
    for channel in enumerate(channels_to_plot):
        plt.subplot(r, c, channel[0] + 1)
        plt.scatter(scales[scale_num].X_hsne[:, 0], scales[scale_num].X_hsne[:, 1],
                    c=scales[scale_num].X[:, channel[0]],
                    s=scales[scale_num].W)
        plt.xlabel('HSNE1')
        plt.ylabel('HSNE2')
        plt.title('Colored by %s' % channel[1])
        plt.colorbar()
    plt.show()

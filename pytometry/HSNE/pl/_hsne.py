import matplotlib.pyplot as plt



def hsne(adata, color=None, scale_num=-1, subplot_grid_dim=(1, 1)):
    try:
        adata.uns['hsne_scales']
    except KeyError as e:
        raise Exception("scales have to be calculated first with tl.hsne")
    scales = adata.uns['hsne_scales']

    settings = adata.uns['hsne_settings']
    all_channel_names = list(adata.var_names.values)
    all_calculated_channel_names = adata.var_names.values[settings['imp_channel_ind']]  # available channels
    if color is None:
        color = all_calculated_channel_names

    for channel_name in color:
        if channel_name not in all_calculated_channel_names:
            raise Exception("Channel with name %s not found in \n%s"%(channel_name, all_calculated_channel_names))

    channels_to_plot_ind = [all_channel_names.index(name) for name in color] #all_calculated_channel_names]


    # if self.scales[scale_num].X_hsne is None:
    #     # embedding for this scale has not been calculated yet
    #     self.scales[scale_num].calc_embedding()

    # dynamically add rows and cols for subplots
    num_channels = len(color)
    r, c = subplot_grid_dim
    if r == -1 or c == -1 or r * c < num_channels:
        while r * c < num_channels:
            if r > c:
                c += 1
            else:
                r += 1
    # create subplots
    plt.figure()
    for channel in enumerate(color):
        plt.subplot(r, c, channel[0] + 1)
        plt.scatter(scales[scale_num].X_hsne[:, 0], scales[scale_num].X_hsne[:, 1],
                    c=scales[scale_num].X[:, channel[0]],
                    s=scales[scale_num].W)
        plt.xlabel('HSNE1')
        plt.ylabel('HSNE2')
        plt.title('Colored by %s' % channel[1])
        plt.colorbar()
    plt.show()

"""
Author:         Felix Hempel
ICB             HelmholtzZentrum München
Creation Date:  05.02.2020

#TODO Description
Gating
"""

# import FlowCytometryTools as fct
from pathlib import Path
from tkinter import filedialog, StringVar

import numpy as np
import anndata
# import converter

# from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from PyQt5.QtX11Extras import xcb_connection_t
from bokeh.transform import transform
from fcswrite import fcswrite
from matplotlib.pyplot import figure
from plotly.graph_objs.layout import xaxis

from tools.FlowCytometryTools.core.bases import Measurement
from pandas import DataFrame
# from pylab import *
from tools.FlowCytometryTools import FCMeasurement, ThresholdGate, PolyGate, QuadGate, test_data_dir, test_data_file
import os

from ipywidgets import widgets
from IPython.display import *
import anndata
import plotly.graph_objects as go
import plotly.offline as offplot
import numpy as np
from flask import Flask, render_template, Markup, request, url_for, redirect
import pytometry
from converter import fcswriter
print('Everything imported')

from ipywidgets import interactive, HBox, VBox

## .fcs files


def get_gate_from_interactive(fcs_file=None, transformed_sample=None):
    if fcs_file is None and transformed_sample is None:
        raise Exception('No .fcs file or sample given')
    else:
        if transformed_sample is None:
            transformed_sample = get_sample_data(fcs_file)
        figure()
        g = transformed_sample.get_gates()
        return g  # does not return anything


# gets sampling data from .fcs file
def get_sample_data(fcs_file):
    if fcs_file is None or (not fcs_file.endswith(".fcs") and not fcs_file.endswith(".h5ad")):
        raise Exception('No .fcs file given')
    else:
        if fcs_file.endswith(".fcs"):
            return FCMeasurement(ID="Data", datafile=fcs_file)
        if fcs_file.endswith(".h5ad"):
            # convert
            adata = anndata.read_h5ad(fcs_file)
            fcswrite.write_fcs(Path('/home/felix/Public/ICB/Backup/KR_full' + '_converted' + '.fcs'),
                               np.array(adata.var_names).tolist(),
                               adata.X, None, 'big', False, False, False, False, False)
            file = r"/home/felix/Public/ICB/Backup/KR_full_converted.fcs"
            return FCMeasurement(ID="Data", data_file=file)


# Returns FCMeasurement
# if file is .h5ad:
# - creates temporary .fcs file
# - creates FCMeasurement
# - deletes temporary .fcs file
def make_sample(file):
    dir = os.path.dirname(file) + "/"
    if file.endswith(".h5ad"):
        tempfile = r"%stemp.fcs" % dir
        # String to avoid duplicate keywords
        clear_dupl = ['__header__', '_channels_', '_channel_names_',
                      '$BEGINANALYSIS', '$ENDANALYSIS', '$BEGINSTEXT', '$ENDSTEXT',
                      '$BEGINDATA', '$ENDDATA', '$BYTEORD', '$DATATYPE',
                      '$MODE', '$NEXTDATA', '$TOT', '$PAR', '$fcswrite version']
        adata = anndata.read_h5ad(file)
        dictionary = adata.uns['meta']
        ch_shortnames = dictionary['_channels_'][:, 0]
        count = 1
        for name in dictionary['_channel_names_']:
            dictionary['$P' + str(count) + 'S'] = name
            count = count + 1
        for i in clear_dupl:
            dictionary.pop(i, None)
        fcswriter.write_fcs(Path(tempfile), ch_shortnames, np.array(adata.var_names).tolist(),
                            adata.X, dictionary, 'big', False)
        return FCMeasurement(ID="Data", datafile=tempfile)

    elif file.endswith(".fcs"):
        return FCMeasurement(ID="Data", datafile=file)

# starts FCT interactive gating GUI
def start_interactive_gating(file):
    dir = os.path.dirname(file) + "/"
    if file.endswith(".h5ad"):
        tempfile = r"%stemp.fcs" % dir
        # String to avoid duplicate keywords
        clear_dupl = ['__header__', '_channels_', '_channel_names_',
                      '$BEGINANALYSIS', '$ENDANALYSIS', '$BEGINSTEXT', '$ENDSTEXT',
                      '$BEGINDATA', '$ENDDATA', '$BYTEORD', '$DATATYPE',
                      '$MODE', '$NEXTDATA', '$TOT', '$PAR', '$fcswrite version']
        adata = anndata.read_h5ad(file)
        dictionary = adata.uns['meta']
        ch_shortnames = dictionary['_channels_'][:, 0]
        count = 1
        for name in dictionary['_channel_names_']:
            dictionary['$P' + str(count) + 'S'] = name
            count = count + 1
        for i in clear_dupl:
            dictionary.pop(i, None)
        fcswriter.write_fcs(Path(tempfile), ch_shortnames, np.array(adata.var_names).tolist(),
                            adata.X, dictionary, 'big', False)
        samp = FCMeasurement(ID="Data", datafile=tempfile)
        
        samp.set_data(samp.data)  # remove for more ram less disk
        samp.datafile = None  # remove for more ram less disk
        os.remove(tempfile)  # remove for more ram less disk
        figure()
        samp.view_interactively()
        # os.remove(tempfile)  # add for more ram less disk
    elif file.endswith(".fcs"):
        samp = FCMeasurement(ID="Data", datafile=file)
        figure()
        samp.view_interactively()


######################################################################################

## Variables
link_fcs = r"/home/felix/Public/VB.fcs"
link_h5ad = r"/home/felix/Public/KR_full_converted.h5ad"





# returns list of gates and corresponding list of points
def get_from_interactive():

    # Gate with parent of the gate
    class GateObject:
        def __init__(self, gate, sample, parent=None):
            self.gate = gate        # this gate
            self.parent = parent    # parent GateObject (if None: root is parent)
            self.sample = sample    # sample of this gate
            self.generation = self.init_generation()

        def init_generation(self):
            if self.parent is None:
                return 0
            else:
                return self.parent.init_generation()+1


    class GatePlotter:

        def __init__(self):
            import tkinter as tk
            self.gates = list()     # first return value:   list of GateLink's
            self.content = list()   # second return value:  X of gated anndata

            self.window = tk.Tk()
            self.filename = ''
            self.root_sample = None  # original sample (kinda backup)
            self.work_sample = None  # sample to be altered

            # BEGIN Developer options #TODO remove
            self.filename = r"/home/felix/Public/KR_full_converted.h5ad"
            # END Developer options

            self.root_sample = make_sample(self.filename)
            self.work_sample = self.root_sample.copy()

            self._filename_var = StringVar()
            self._filename_var.set(self.filename)  # Variable for label_fileName

            # Declare widgets
            self.button_selectFile = tk.Button(master=self.window, text="Select a file",
                                               command=lambda: self.get_file_from_dir())
            self.label_fileName = tk.Label(master=self.window, textvariable=self._filename_var)
            self.button_startGating = tk.Button(master=self.window, text='Start gating',
                                                command=lambda: self.start_gating(self.work_sample))
            self.listBox_gates = tk.Listbox(master=self.window)  # , selectmode='multiple'
            self.button_applyGates = tk.Button(master=self.window, text="Apply selected gates",
                                               command=lambda: self.apply_gates())

            # Pack widgets in window
            self.button_selectFile.pack()
            self.label_fileName.pack()
            self.button_startGating.pack()
            self.listBox_gates.pack()
            self.button_applyGates.pack()

            # Bindings
            self.listBox_gates.bind('<Double-Button-1>', self.open_gate)

            # Start main loop
            self.window.mainloop()


        def _transform_sample(self, sample, function='hlog', direction='forward'):
            '''
            Transforms self.work_sample
            :param function: hlog|tlog
            :param direction: forward|inverse
            :return transformed sample <FCMeasurement>
            '''
            if function is 'hlog':
                for channel in sample.channel_names:
                    if channel is not 'Time':  #except time graphs
                        sample = sample.transform('hlog', channels=channel, direction=direction, b=1, r=100, d=100)
            if function is 'tlog':
                for channel in sample.channel_names:
                    if channel is not 'Time':  #except time graphs
                        sample = sample.transform('tlog', channels=channel, direction=direction, th=1)
            return sample


        def start_gating(self, sample=None):
            #import tkinter as tk
            from tkinter import END
            if sample is None:
                sample = self.root_sample
            sample = self._transform_sample(sample, direction='forward')
            _gates_temp = get_gate_from_interactive(transformed_sample=sample)
            for g in _gates_temp:
                gobj = GateObject(g, sample.gate(g))
                gobj.gate.name = '%d %s' %(gobj.generation,gobj.gate.name)
                self.gates.append(gobj)
                #self.content.append(self.get_content(g))
                self.listBox_gates.insert(END, gobj.gate.name)

        def get_content(self, gate):
            '''
            Gates sample and returns new sample
            :param gate: gate
            :return: gated sample
            '''


        def apply_gates(self):
            for selected_gate in self.listBox_gates.curselection():
                for gate_obj in self.gates:
                    if gate_obj.gate.name == self.listBox_gates.get(selected_gate):
                        self.work_sample = self.work_sample.gate(gate_obj.gate)


        def get_file_from_dir(self):
            '''
            Opens file system and filters for .fcs and .h5ad files
            '''
            temp_filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                       filetypes=(("anndata files", "*.h5ad"), ("fcs files", "*.fcs*")))
            if temp_filename is not None:
                # show selected filename in label
                self.filename = r'%s' % temp_filename
                self._filename_var.set(self.filename)
                # create sample
                self.root_sample = make_sample(self.filename)
                self.work_sample = self.root_sample.copy()

        def open_gate(self, event):
            '''
            Not implemented yet TODO implement
            :param event: Double click event
            :return:
            '''
            from tkinter import END
            curgate_name = self.listBox_gates.curselection()
            print('Curselection: %s'%curgate_name)
            for gate_obj in self.gates:
                if gate_obj.gate.name == self.listBox_gates.get(curgate_name):
                    print('GateObject: %s' %gate_obj)
                    print('...and its parent: %s' %gate_obj.parent)
                    new_gates = get_gate_from_interactive(transformed_sample=gate_obj.sample)
                    for ng in new_gates:
                        ngobj = GateObject(ng, gate_obj.sample.gate(ng), parent=gate_obj)
                        ngobj.gate.name = '%d %s' %(ngobj.generation,ngobj.gate.name)
                        self.gates.append(ngobj)
                        self.listBox_gates.insert(END, ngobj.gate.name)


    gp = GatePlotter()




def gateProg_with_gatePlotter():
    # START Plotly GatePlotter
    import plotly.graph_objects as go
    import numpy as np


    class Gate_Plotter:
        def __init__(self, adata):
            self.adata = adata
            self.fig = None
            self.gate_dict = dict()
            for c1 in self.adata.var_names:
                for c2 in self.adata.var_names:
                    self.gate_dict.update({'%s %s' % (c1, c2): list()})

        def show(self, xchannel=0, ychannel=1):
            self.fig = go.Figure()
            self.fig.add_trace(go.Scattergl(
                x=adata.X[:, xchannel],  # list(adata.var_names).index(xchannel) #if name
                y=adata.X[:, ychannel],
                mode='markers',
                marker=dict(color='rgba(0, 0, 0, 0.8)', size=0.9)
            )
            )
            # add gates as scatter traces
            channel_name_x = list(self.adata.var_names)[xchannel]
            channel_name_y = list(self.adata.var_names)[ychannel]
            for g in self.gate_dict['%s %s' % (channel_name_x, channel_name_y)]:
                self.fig.add_trace(g)

            # x-Axis
            self.fig.update_xaxes(type='log')
            self.fig.update_xaxes(title=adata.var_names[xchannel])
            # y-Axis
            self.fig.update_yaxes(type='log')
            self.fig.update_yaxes(title=adata.var_names[ychannel])
            self.fig.show()

        def add_gate(self, gatetype, verts, region, channels, name):
            if gatetype == 'poly':
                # TODO: region (currently just 'in')
                # add to gate_dict
                self.gate_dict['%s %s' % (channels[0], channels[1])].append(
                    go.Scatter(x=verts[:, 0], y=verts[:, 1], fill='toself', mode='markers', name=name))
                self.gate_dict['%s %s' % (channels[1], channels[0])].append(
                    go.Scatter(x=verts[:, 1], y=verts[:, 0], fill='toself', mode='markers'))

            if gatetype == 'thresh':
                ma = max(adata.X[:, list(adata.var_names).index(channels[0])])  # max value on channel axis for visual gate
                mi = min(adata.X[:, list(adata.var_names).index(channels[0])])  # min value on channel axis for visual gate
                for comb in [d for d in self.gate_dict]:
                    if comb.startswith(channels[0]):  # unten oben
                        if region is 'below':
                            self.gate_dict[comb].append(
                                go.Scatter(x=[ma, mi], y=[verts[0], verts[0]], fill='tozeroy', mode='markers', name=name))
                        if region is 'above':
                            # self.gate_dict[comb].append(go.Scatter(x=[mi,ma,ma,mi], y=[verts[0],verts[0],], fill='toself', mode='markers', name=name))
                            pass
                    elif channels[0] in comb:  # links rechts
                        if region is 'below':
                            self.gate_dict[comb].append(
                                go.Scatter(x=[verts[0], verts[0]], y=[0, ma], fill='tozerox',  name=name))
                        if region is 'above':
                            # TODO

                            pass
                    # self.gate_dict[channels[0]].append(go.Scatter(x=))


    # reading Anndata File
    print('Reading anndata file...')
    adata = anndata.read_h5ad(link_h5ad)
    obs = adata.obs
    print('...done')

    channel_1 = 'FSC-A'
    channel_2 = 'FSC-H'

    gp = Gate_Plotter(adata)
    # (self, gatetype, verts, region, channels, name)
    gp.add_gate('poly', np.array([(800000, 400000), (1200000, 800000), (100000, 900000)]), None, [channel_1, channel_2], 'Gate1')
    #gp.add_gate('poly', np.array([(100, 200), (17000, 100), (100000, 40000)]), None, [channel_1, channel_2], 'Gate2')
    #gp.add_gate('thresh', [100000], 'below', [channel_1], 'ThreshGate')

    gp.show(xchannel=list(adata.var_names).index(channel_1), ychannel=list(adata.var_names).index(channel_2))
    gp.show(xchannel=list(adata.var_names).index(channel_2), ychannel=list(adata.var_names).index(channel_1))
    #END Plotly GatePlotter


def gateProg_with_GUI():
    import tkinter as tk

    class gating_GUI():
        def __init__(self):
            self.window = tk.Tk()
            self.filename = ''
            self.orig_sample = None     # original sample (kinda backup)
            self.work_sample = None     # sample to be altered

            # BEGIN Developer options #TODO remove
            self.filename = r"/home/felix/Public/KR_full_converted.h5ad"
            # END Developer options

            self.orig_sample = make_sample(self.filename)
            self.work_sample = self.orig_sample.copy()
            #transform
            self._transform_work_sample(function='hlog')

            self._filename_var = StringVar()
            self._filename_var.set(self.filename)  # Variable for label_fileName
            self.gates = list()

            # Declare widgets
            self.button_selectFile = tk.Button(master=self.window,text="Select a file", command=lambda: self.get_file_from_dir())
            self.label_fileName = tk.Label(master=self.window, textvariable=self._filename_var)
            self.button_startGating = tk.Button(master=self.window, text='Start gating', command=lambda: self.start_gating())
            self.listBox_gates = tk.Listbox(master=self.window) #, selectmode='multiple'
            self.button_applyGates = tk.Button(master=self.window, text="Apply selected gates", command=lambda: self.apply_gates())


            # Pack widgets in window
            self.button_selectFile.pack()
            self.label_fileName.pack()
            self.button_startGating.pack()
            self.listBox_gates.pack()
            self.button_applyGates.pack()

            # Bindings
            self.listBox_gates.bind('<Double-Button-1>', self.open_interactive_gate_view)

            # Start main loop
            self.window.mainloop()


        def open_interactive_gate_view(self, event):
            self.work_sample = self.orig_sample.copy()
            # transform
            self._transform_work_sample(function='hlog')
            for gate in self.gates:
                if gate.name == self.listBox_gates.get(self.listBox_gates.curselection()):
                    self.work_sample = self.work_sample.gate(gate)
                    g = get_gate_from_interactive(transformed_sample=self.work_sample) #TODO somehow add gates

        # Transforms
        def _transform_work_sample(self, function='flog'):
            if function is 'hlog':
                for channel in self.work_sample.channel_names:
                    if channel is not 'Time':  #except time graphs
                        self.work_sample = self.work_sample.transform('hlog', channels=channel, b=1, r=100, d=100)
            if function is 'tlog':
                for channel in self.work_sample.channel_names:
                    if channel is not 'Time':  #except time graphs
                        self.work_sample = self.work_sample.transform('tlog', channels=channel, th=1)
            if function is 'flog':
                for channel in self.work_sample.channel_names:
                    if channel is not 'Time':
                        ind = self.work_sample.channel_names.index(channel)
                        print(self.work_sample.data.values[0,0])
                        print(np.min(self.work_sample.data.values[:,ind]))
                        self.work_sample.data.values[:,ind] += np.min(self.work_sample.data.values[:,ind])
                        print(self.work_sample.data.values[0,0])
                        print(np.min(self.work_sample.data.values[:,ind]))



        def get_file_from_dir(self):
            temp_filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                       filetypes=(("anndata files", "*.h5ad"), ("fcs files", "*.fcs*")))
            if temp_filename is not None:
                # show selected filename in label
                self.filename = r'%s' % temp_filename
                self._filename_var.set(self.filename)
                # create sample
                self.orig_sample = make_sample(self.filename)
                self.work_sample = self.orig_sample.copy()


        def start_gating(self):
            #sample = make_sample(self.filename)
            _gates_temp = get_gate_from_interactive(transformed_sample=self.work_sample)
            for g in _gates_temp:
                self.gates.append(g)
                self.listBox_gates.insert(tk.END, g.name)

        def apply_gates(self):
            for selected_gate in self.listBox_gates.curselection():
                for gate in self.gates:
                    if gate.name == self.listBox_gates.get(selected_gate):
                        self.work_sample = self.work_sample.gate(gate)





    gg = gating_GUI()



## Test Method: gating with wxpython
# Current state: Does not work due to wxpython package import difficulties
def gateProg_with_wx():


    sample = make_sample(link_h5ad)
    gates = get_gate_from_interactive(transformed_sample=sample)
    #gates = sample.get_gates()
    #gates = make_sample(link_h5ad).get_gates()

    ## apply gates
    # TODO all crap
    figure()
    gate1 = ThresholdGate(4.909e+07, ('FSC-A'), region='above', name='gate1')   # region: above/below
    gate2 = ThresholdGate(8.931e+05, ('FSC-H'), region='below', name='gate2')   # region: above/below
    gate3 = PolyGate([(2.236e+07, 2.216e+06), (5.835e+06, 2.209e+06), (7.321e+06, 2.745e+06), (2.811e+07, 2.792e+06)],
                     ('FSC-A', 'FSC-H'), region='in', name='gate1')          # region: in/out
    gate4 = QuadGate((3.981e+07, 1.081e+06), ('FSC-A', 'FSC-H'),
                     region='bottom right', name='gate4')                    # region: 'top left', 'top right', 'bottom left', 'bottom right'

    # alter gates TODO: apply multiple gates at once
    if gates is not None:
        for g in gates:
            sample = sample.gate(g)
            print(type(g))
        #sample = sample.gate(gates)
        sample.view_interactively()

    # Channels to plot
    channel1 = sample.channel_names[2]
    channel2 = sample.channel_names[10]

    # Transformation
    #tsample500 = sample.transform('hlog', channels=[channel1, channel2], b=500.0)
    # tsample100 = sample.transform('hlog', channels=[channel1, channel2], b=100.0)
    # tsample1000 = sample.transform('hlog', channels=[channel1, channel2], b=1000.0)

    # adata --> measurement
    # TODO: channels not compatible
    adata = anndata.read_h5ad(link_h5ad)  # anndata object
    v = adata.var

    meat = adata.uns['meta']
    meat['_channel_names_'] = tuple(meat['_channel_names_'])
    channnn = adata.uns['channels']
    meat['_channels_'] = adata.uns['channels']

    print("Channels")
    print(meat['_channels_'])
    print("Channel names")
    print(meat['_channel_names_'])

    mes = Measurement(ID="Normal Measurement", metafile=meat)
    mes.data = adata._get_X()
    mes.meta = adata.uns['meta']
    mes.channels = adata.uns['channels']
    mes.channel_names = tuple(adata.var.T.columns)

    FCmes = FCMeasurement(ID="FC Measurement", readmeta=False, metafile=meat)
    FCmes.meta = adata.uns['meta']
    FCmes.data = DataFrame(adata._get_X())

    # FCmes.__setattr__('channels', adata.uns['channels'])
    # FCmes.channels = adata.uns['channels']
    # FCmes.__setattr__('channel_names', tuple(adata.var.T.columns))
    # FCmes.channel_names = tuple(adata.var.T.columns)

    print('Test measurement created')

    ## TODO interactive window
    # app = App(False)
    print('Starting interactive')
    ## h5ad data gating
    # ax = plt.scatter(adata._get_X()[:, channel1], adata._get_X()[:, channel2])
    # polsel = PolygonSelector(ax, onselect=onselect)

    ## FCS data gating
    gate = get_gate_from_interactive(transformed_sample=FCmes)
    # gate2 = get_gate_from_interactive(data_file, tsample500)
    print('Interactive closed')

    print("Done")

get_from_interactive()
#gateProg_with_GUI()
#gateProg_with_gatePlotter()
#gateProg_with_wx()
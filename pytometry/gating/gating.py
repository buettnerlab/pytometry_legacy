import os
import anndata
#from prompt_toolkit.filters import emacs_selection_mode

from pytometry.converter import fcswriter
from pathlib import Path
import numpy as np
from pytometry.tools.FlowCytometryTools import FCMeasurement
from tkinter import filedialog, StringVar
from matplotlib.pyplot import figure
from fcswrite import fcswrite



# Help functions

def make_sample(file):
    '''
    Returns FCMeasurement (sample) of given file
    :param file:
        if file is .h5ad:
            - creates temporary .fcs file
            - creates FCMeasurement (sample)
            - deletes temporary .fcs file
        if file is .fcs:
            - creates FCMeasurement (sample)
    :return: FCMeasurement (sample)
    '''
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


def get_gate_from_interactive(fcs_file=None, sample=None):
    '''
    opens interactive gating GUI
    :param fcs_file: .fcs file
    :param sample:
    :return: list of gates (g)
    '''
    if fcs_file is None and sample is None:
        raise Exception('No .fcs file or sample given')
    else:
        if sample is None:
            sample = get_sample_data(fcs_file)
        figure()
        g = sample.get_gates()
        return g

#TODO
def get_sample_data(fcs_file):
    '''
    Gets sampling data from .fcs file
    :param fcs_file: .fcs file
    :return: FCMeasurement (sample)
    '''
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
                               adata.X, {}, 'big', False, False, False, False, False)
            file = r"/home/felix/Public/ICB/Backup/KR_full_converted.fcs"
            return FCMeasurement(ID="Data", data_file=file)


# Gate with parent of the gate
class GateObject:
    def __init__(self, gate, sample, parent=None):
        self.gate = gate        # this gate
        self.parent = parent    # parent GateObject (if None: root is parent)
        self.sample = sample    # sample of this gate
        self.generation = self.init_generation()
        #TODO wieviele Zellen

    def init_generation(self):
        if self.parent is None:
            return 0
        else:
            return self.parent.init_generation()+1


class GatePlotter:

    def __init__(self):
        import tkinter as tk
        from tkinter.ttk import Treeview
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
        self.treeView_gates = Treeview(master=self.window, selectmode="browse") #Treeview
        self.button_addGate = tk.Button(master=self.window, text="Add gate",
                                        command=lambda: self.open_gate(None))
        self.button_deleteGate = tk.Button(master=self.window, text="Delete gate",
                                           command=lambda: self.btn_deleteGate())

        # Pack widgets in window
        self.button_selectFile.pack()
        self.label_fileName.pack()
        self.button_startGating.pack()
        self.treeView_gates.pack()
        self.button_addGate.pack()
        self.button_deleteGate.pack()

        # Bindings
        self.treeView_gates.bind('<Double-Button-1>', self.open_gate)

        # Start main loop
        self.window.mainloop()


    def _refreshTree(self):
        #sort by "level" (root first)
        self.treeView_gates.delete(*self.treeView_gates.get_children())
        queue = self.gates.copy()
        while len(queue) is not 0:
            for g in queue:
                if self.treeView_gates.exists(g.gate.name):
                    queue.remove(g)
                else:
                    if g.parent is None:
                        self.treeView_gates.insert('','end',g.gate.name, text=g.gate.name)
                        queue.remove(g)
                    else:
                        if self.treeView_gates.exists(g.parent.gate.name):
                            self.treeView_gates.insert(g.parent.gate.name, 'end', g.gate.name, text=g.gate.name)
                            queue.remove(g)


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

    def btn_deleteGate(self):
        curgate = self.treeView_gates.focus()
        if curgate is not None:
            # get parent of doomed gate
            parent = None
            for g in self.gates:
                if g.gate.name == curgate:
                    if g.parent is None:
                        parent = ''
                    else:
                        parent = g.parent.gate.name
                    self.gates.remove(g)
            for g in self.gates:
                if g.parent is not None:
                    if g.parent.gate.name == curgate:
                        g.parent.gate.name = parent

        self._refreshTree()


    def start_gating(self, sample=None):
        if sample is None:
            sample = self.root_sample
        sample = self._transform_sample(sample, direction='forward')
        _gates_temp = get_gate_from_interactive(sample=sample)
        if _gates_temp is not None:
            for g in _gates_temp:
                g.name = self.validate_gate_name(g.name)
                gobj = GateObject(g, sample.gate(g))
                self.gates.append(gobj)
                #self.content.append(self.get_content(g))
        self._refreshTree()

    def validate_gate_name(self, name):
        blacklist = [g.gate.name for g in self.gates]
        if name in blacklist:
            i = 1
            ext = ' (%d)'%i
            while name+ext in blacklist:
                i = i + 1
                ext = ' (%d)'%i
            return name+ext
        return name


    def get_file_from_dir(self):
        '''
        Opens file system and filters for .fcs and .h5ad files
        '''
        temp_filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                   filetypes=([("anndata files", "*.h5ad"), ("fcs files", "*.fcs")]))
        if temp_filename is not None:
            # show selected filename in label
            self.filename = r'%s' % temp_filename
            self._filename_var.set(self.filename)
            # create sample
            self.root_sample = make_sample(self.filename)
            self.work_sample = self.root_sample.copy()

    def open_gate(self, event):
        '''

        :param event: Double click event
        :return:
        '''
        from tkinter import END
        curgate_name = self.treeView_gates.focus()
        print('Curselection: %s'%curgate_name)
        for gate_obj in self.gates:
            if gate_obj.gate.name == curgate_name:
                print('GateObject: %s' %gate_obj)
                print('...and its parent: %s' %gate_obj.parent)
                new_gates = get_gate_from_interactive(sample=gate_obj.sample)
                if new_gates is not None:
                    for ng in new_gates:
                        ngobj = GateObject(ng, gate_obj.sample.gate(ng), parent=gate_obj)
                        ngobj.gate.name = self.validate_gate_name(ngobj.gate.name)
                        self.gates.append(ngobj)
        self._refreshTree()


gp = GatePlotter()
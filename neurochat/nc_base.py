# -*- coding: utf-8 -*-
"""
This module implements two classes NAbstract and NBase.

Those are inherited by other data classes for detailed implementation.
Methods and attributes that are likely to be common in other data types in
NeuroChaT are implemented in the NAbstract and NBase classes.

@author: Md Nurul Islam; islammn at tcd dot ie

"""
import inspect
import logging

from collections import OrderedDict as oDict


class NAbstract(object):
    """
    NAbstract includes a number of attributes and methods common in NC.

    NAbstract is the abstract class which includes a number of
    attributes and methods commonly used by most other data types.

    """

    def __init__(self, **kwargs):
        """
        Instantiate the `NAbstract` class.

        Parameters
        ---------
        **kwargs
            Keyword arguments, directly supported are:
            filename, system, and name

        Returns
        -------
        None

        """
        self._filename = kwargs.get('filename', '')
        self._system = kwargs.get('system', 'Axona')
        self._name = kwargs.get('name', 'c0')
        self._description = ''
        self._results = oDict()
        self._record_info = {'File version': '',
                             'Date': '',
                             'Time': '',
                             'Experimenter': '',
                             'Comments': '',
                             'Duration': 0,
                             'Format': 'Axona',
                             'Source': self._filename}

        self.__type = 'abstract'

    def get_type(self):
        """
        Return the type of data class.

        For example, an instance of Nabstract will return
        'abstract' as the type of the class.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self.__type

    def set_filename(self, filename=None):
        """
        Set the file name of the data object.

        Parameters
        ----------
        filename : str
            Name of the data file

        Returns
        -------
        None

        """
        if filename is not None:
            self._filename = filename

    def get_filename(self):
        """
        Return the filename of the data class.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self._filename

    def set_system(self, system=None):
        """
        Set the name of the recording system or the format of the data file.

        Parameters
        ----------
        system : str
            Recording system or data file format

        Returns
        -------
        None

        """
        if system is not None:
            self._system = system

    def get_system(self):
        """
        Return the name of the recording system or data format.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self._system

    def set_name(self, name=''):
        """
        Set a name for the class instance.

        Parameters
        ----------
        name : str

        Returns
        -------
        None

        """
        self._name = name

    def get_name(self):
        """
        Get the name of the object.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self._name

    def set_description(self, description=''):
        """
        Set the general description about the data by the user.

        Parameters
        ----------
        description : str

        Returns
        -------
        None

        """
        self._description = description

    def save_to_hdf5(self, parent_dir):
        """Only implemented in subclasses."""
        pass  # implement for each type

    def load(self):
        """Only implemented in subclasses."""
        pass

    @classmethod
    def _new_instance(cls, obj=None, **kwargs):
        """
        Create a new instance from the class `cls`.

        If `obj` is None, a new instance of `cls` is returned.
        If the `obj` is an instance of `cls`, the same is returned.
        If `obj` itself is a class, it supersedes the `cls` and
        returns an object of `obj` class.

        Parameters
        ----------
        cls
            Class of the new node
        obj
            Either an object of class `cls` or a Class to be instantiated

        Returns
        -------
            New node of specified Class instance

        """
        if obj is None:
            new_obj = cls(**kwargs)
        elif isinstance(obj, cls):
            new_obj = obj
        elif inspect.isclass(obj):
            cls = obj
        new_obj = cls(**kwargs)

        return new_obj

    def get_results(self):
        """
        Return the analysis results.

        Returns
        -------
        OrderedDict

        """
        return self._results

    def update_result(self, new_result={}):
        """
        Update the results.

        Parameters
        ----------
        new_result : dict
            The dictionary to update the current results dict with.

        Returns
        -------
        OrderedDict

        See Also
        --------
        get_results

        """
        self._results.update(new_result)

    def reset_results(self):
        """Reset the results to an empty OrderedDict."""
        self._results = oDict()

    def _set_file_version(self, version=''):
        """
        Set the file version as decoded from the native formats.

        Parameters
        ----------
        version : str

        Returns
        -------
        None

        """
        self._record_info['File version'] = version

    def _set_date(self, date_str=''):
        """
        Set the date of the experiment.

        Parameters
        ----------
        date_str : str

        Returns
        -------
        None

        """
        self._record_info['Date'] = date_str

    def _set_time(self, time=''):
        """
        Set the time of the experiment.

        Parameters
        ----------
        time : str

        Returns
        -------
        None

        """
        self._record_info['Time'] = time

    def _set_experimenter(self, experimenter=''):
        """
        Set the name of the experimenter.

        Parameters
        ----------
        experimenter : str

        Returns
        -------
        None

        """
        self._record_info['Experimenter'] = experimenter

    def _set_comments(self, comments=''):
        """
        Set comments or notes of the experiment.

        Parameters
        ----------
        comments : str

        Returns
        -------
        None

        """
        self._record_info['Comments'] = comments

    def _set_duration(self, duration=''):
        """
        Set the duration of the experiment.

        Parameters
        ----------
        duration : float

        Returns
        -------
        None

        """
        self._record_info['Duration'] = duration

    def _set_source_format(self, system='Axona'):
        """
        Set the recording format or the source-format of the data.

        Parameters
        ----------
        system: str

        Returns
        -------
        None

        """
        self._record_info['Format'] = system

    def _set_data_source(self, filename=None):
        """
        Set the source of the original data file.

        Parameters
        ----------
        filename : str

        Returns
        -------
        None

        """
        self._record_info['Source'] = filename

    def get_file_version(self):
        """
        Get the version of the data file.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self._record_info['File version']

    def get_date(self):
        """
        Get the recording date.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self._record_info['Date']

    def get_time(self):
        """
        Get the time of the experiment.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self._record_info['Time']

    def get_experimenter(self):
        """
        Get the name of the experimenter.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self._record_info['Experimenter']

    def get_comments(self):
        """
        Get the comments or notes about the experiment.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self._record_info['Comments']

    def get_duration(self):
        """
        Get the duration of the experiment.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self._record_info['Duration']

    def get_source_format(self):
        """
        Get the recording system or native data format.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self._record_info['Format']

    def get_data_source(self):
        """
        Get the source of the data.

        Parameters
        ----------
        None

        Returns
        -------
        str

        """
        return self._record_info['Source']

    def set_record_info(self, new_info={}):
        """
        Set the recording information.

        Parameters
        ----------
        new_info: dict
            The new recording information dictionary.
            The existing information is updated with this information.

        Returns
        -------
        None

        """
        self._record_info.update(new_info)

    def get_record_info(self, record_name=None):
        """
        Get the comments or notes about the experiment.

        Parameters
        ----------
        None

        Returns
        -------
        dict or str
            dict if record_name is None.
            str if record_name is a key in _record_info.

        """
        if record_name is None:
            return self._record_info
        else:
            return self._record_info.get(record_name, None)


class NBase(NAbstract):
    """
    NBase adds managing multiple spike or LFP datasets to NAbstract.

    NBase is derived from NAbstract class. And it implements additional
    functionalities for managing multiple spike or LFP datasets to
    NAbstract.

    """

    def __init__(self, **kwargs):
        """
        Instantiate the `NBase` class.

        Parameters
        ---------
        **kwargs
            Keyword arguments

        """
        super().__init__(**kwargs)
        self._spikes = []
        self._spikes_by_name = oDict()
        self._lfp = []
        self._lfp_by_name = oDict()

        self._record_info = {'File version': '',
                             'Date': '',
                             'Time': '',
                             'Experimenter': '',
                             'Comments': '',
                             'Duration': 0,
                             'No of channels': 1,
                             'Bytes per timestamp': 1,
                             'Sampling rate': 1,
                             'Bytes per sample': 1,
                             'ADC Fullscale mv': 1,
                             'Format': 'Axona',
                             'Source': self._filename}
        self.__type = 'base'

    def add_node(self, node, node_type=None, **kwargs):
        """
        Add a new dataset, called node, to the spike and LFP dataset arrays.

        Parameters
        ----------
        node
            Data node to be added
        node_type : str
            Type of the dataset described in each class attributes
        **kwargs
            Keyword arguments

        Returns
        -------
        None

        """
        name = node.get_name()
        _replace = kwargs.get('replace', False)

        if node_type is None:
            logging.error('Node type is not defined')
        elif node_type == 'spike':
            node_names = self.get_spike_names()
            nodes = self._spikes
            nodesByName = self._spikes_by_name
        elif node_type == 'lfp':
            node_names = self.get_lfp_names()
            nodes = self._lfp
            nodesByName = self._lfp_by_name

        if _replace:
            i = self.del_node(node)
        elif name in node_names:
            logging.warning(
                node_type +
                ' with name {0} already exists, '.format(name) +
                'cannot add another one.\r\n' +
                'Try renaming or set replace True')
        else:
            i = len(nodes)

        nodes.insert(i, node)
        nodesByName[name] = node

    def del_node(self, node):
        """
        Delete a node that represents spike or LFP dataset.

        Parameters
        ----------
        node
            Data node to be deleted

        Returns
        -------
        int
            Index of deleted node

        """
        i = None
        if node in self._spikes:
            i = self._spikes.index(node)
            self._spikes.remove(node)
            del self._spikes_by_name[node.get_name()]
        elif node in self._lfp:
            i = self._lfp.index(node)
            self._lfp.remove(node)
            del self._lfp_by_name[node.get_name()]
        return i

    def get_node(self, node_names, node_type='spike'):
        """
        Get the nodes by name and dataset type.

        Parameters
        ----------
        node_names : list
            List of the names of the data nodes to obtain
        node_type : str
            Type of the data node

        Returns
        -------
        list
            List of the data nodes

        """
        nodes = []
        not_nodes = []
        if node_type == 'spike':
            names = self.get_spike_names()
            nodes = self._spikes_by_name
        elif node_type == 'lfp':
            names = self.get_lfp_names()
            nodes = self._lfp_by_name
        for name in node_names:
            nodes.append(nodes[name]) if name in names\
                else not_nodes.append(name)
        if not_nodes:
            logging.warning(','.join(not_nodes) + ' does not exist')
        return nodes

    def get_spike(self, names=None):
        """
        Get the spike nodes by name.

        Parameters
        ----------
        names : list
            List of the names of the spike nodes to obtain

        Returns
        -------
        list
            List of the spike nodes.
            Returns all the spike nodes if `names` is None.

        """
        if names is None:
            spikes = self._spikes
        else:
            spikes = self.get_node(names, 'spike')
        return spikes

    def get_lfp(self, names=None):
        """
        Get the lfp nodes by name.

        Parameters
        ----------
        names : list
            List of the names of the lfp nodes to obtain

        Returns
        -------
        list
            List of the lfp nodes. Returns all the lfp nodes if `names` is None

        """
        if names is None:
            lfp = self._lfp
        else:
            lfp = self.get_node(names, 'lfp')
        return lfp

    def del_spike(self, spike):
        """
        Delete a node that represents a spike dataset.

        Parameters
        ----------
        spike
            Spike node to be deleted by name or the object

        Returns
        -------
        i : int
            Index of the deleted node

        """
        if isinstance(spike, str):
            name = spike
            spike = self.get_spike(name)
        i = self.del_node(spike)

        return i

    def del_lfp(self, lfp):
        """
        Delete a node that represents LFP dataset.

        Parameters
        ----------
        lfp
            LFP node to be deleted by name or the object

        Returns
        -------
        int
            Index of the deleted node

        """
        i = 0
        if isinstance(lfp, str):
            name = lfp
            lfp = self.get_lfp(name)
        i = self.del_node(lfp)

        return i

    def get_spike_names(self):
        """
        Get the names of all the spike nodes.

        Parameters
        ----------
        None

        Returns
        -------
        list
            Names of the spike nodes

        """
        return self._spikes_by_name.keys()

    def get_lfp_names(self):
        """
        Get the name of all the lfp nodes.

        Parameters
        ----------
        None

        Returns
        -------
        list
            Names of the LFP nodes

        """
        return self._lfp_by_name.keys()

    def change_names(self, old_names, new_names, node_type='spike'):
        """
        Change the names of nodes.

        `old_names` should have the same length as that of `new_length`

        Parameters
        ----------
        old_names : list of str
            List of the old names of nodes
        new_names : list of str
            List of the new names of nodes
        node_type
            Type of the data node

        Returns
        -------
        None

        """
        if len(new_names) != len(old_names):
            logging.error('Input names are not equal in numbers!')
        elif len(set(new_names)) < len(old_names):
            logging.error('Duplicate names are not allowed!')
        else:
            if node_type == 'spike':
                for i, name in enumerate(new_names):
                    node = self.get_spike(old_names[i])
                    node.set_name(name)
                    self._spikes_by_name[name] = self._spikes_by_name.pop(
                        old_names[i])

            elif node_type == 'lfp':
                for i, name in enumerate(new_names):
                    node = self.get_lfp(old_names[i])
                    node.set_name(name)
                    self._lfp_by_name[name] = self._spikes_by_name.pop(
                        old_names[i])

    def set_spike_names(self, names):
        """
        Set the names of the spike nodes. Old names are replaced.

        Parameters
        ----------
        names : list of str
            List of new names of the spike nodes

        Returns
        -------
        None

        """
        self.change_names(self.get_spike_names(), names, 'lfp')

    def set_lfp_names(self, names):
        """
        Set the names of the lfp nodes. Old names are replaced.

        Parameters
        ----------
        names : list of str
            List of new names of the lfp nodes

        Returns
        -------
        None

        """
        self.change_names(self.get_lfp_names(), names, 'lfp')

    def set_node_file_names(self, node_names, filenames, node_type='spike'):
        """
        Set the filenames for each data node.

        `node_names` must be of equal length to `filenames`

        Parameters
        ----------
        node_names : list of str
            Names of the nodes whose filenames are set
        filenames : list of str
            List of the filenames for each node
        node_type
            Type of the data node

        Returns
        -------
        None

        """
        if len(node_names) != len(filenames):
            logging.error('No. of names does not match with no. of filenames')
        elif len(set(node_names)) != len(node_names):
            logging.error('Duplicate names are not allowed!')
        else:
            nodes = self.get_node(node_names, node_type)
            for node in nodes:
                node.set_filename(filenames(nodes.index(node)))

    def set_spike_file_names(self, spike_names, filenames):
        """
        Set the filenames for each data node.

        `spike_names` must be of equal length to `filenames`

        Parameters
        ----------
        spike_names : list of str
            Names of the spike nodes whose filenames are set
        filenames : list of str
            List of the filenames for each spike node

        Returns
        -------
        None

        """
        self.set_node_file_names(spike_names, filenames, 'spike')

    def set_lfp_file_names(self, lfp_names, filenames):
        """
        Set the filenames for each LFP data node.

        `lfp_names` must be of equal length to `filenames`

        Parameters
        ----------
        lfp_names : list of str
            Names of the lfp nodes whose filenames are set
        filenames : list of str
            List of the filenames for each lfp node

        Returns
        -------
        None

        """
        self.set_node_file_names(lfp_names, filenames, 'lfp')

    def count_spike(self):
        """
        Count the number of spike nodes.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Total number of spike nodes

        """
        return len(self._spikes)

    def count_lfp(self):
        """
        Count the number of lfp nodes.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Total number of lfp nodes

        """
        return len(self._lfp)

    def _add_node(self, cls, node, node_type, **kwargs):
        """
        Add a node instance of class `cls` from `node` in list of `node_type`.

        Existing nodes can be replaced by input `replace= True`

        Parameters
        ----------
        cls
            Class of node to be added
        node
            Either an object of `cls` or a Class.
            If None, a new instance of `cls`
            is added to the node list and returned
        node_type : str
            Type of the data node


        Returns
        -------
            Newly added data node

        See also
        --------
        add_node

        """
        new_node = self._new_instance(node, **kwargs)
        self.add_node(new_node, node_type,
                      replace=kwargs.get('replace', False))

        return new_node

    def _get_instance(self, cls, node, node_type):
        """
        Create node instance of class `cls` from `node` in list of `node_type`.

        Parameters
        ----------
        cls
            Class of node to be added
        node
            Either an object of `cls`or or a Class.
            If None, new instance of `cls` is returned.
        node_type : str
            Type of the data node

        Returns
        -------
            Newly added data node

        See also
        --------
        add_node

        """
        if isinstance(node, cls):
            new_node = node
        else:
            if node_type == 'lfp':
                _get_node_names = self.get_lfp_names
                _get_node = self.get_lfp
            if node_type == 'spike':
                _get_node_names = self.get_spike_names
                _get_node = self.get_spike
            if node in _get_node_names():
                new_node = _get_node(node)

        return new_node

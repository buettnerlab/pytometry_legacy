'''
sinfo finds and prints version information for loaded modules in the current
session, Python, and the OS.
'''

import sys
import types
import platform
import inspect
from datetime import datetime
from importlib import import_module
from inspect import cleandoc
from multiprocessing import cpu_count
from pathlib import Path

from stdlib_list import stdlib_list


def _imports(environ):
    '''Find modules in an environment.'''
    for name, val in environ:
        # If the module was directly imported
        if isinstance(val, types.ModuleType):
            yield val.__name__
        # If something was imported from the module
        else:
            try:
                yield val.__module__.split('.')[0]
            except AttributeError:
                pass


def _find_version(mod_version_attr):
    '''Find the version number of a module'''
    if (isinstance(mod_version_attr, str)
            or isinstance(mod_version_attr, int)):
        return mod_version_attr
    elif isinstance(mod_version_attr, tuple):
        joined_tuple = '.'.join([str(num) for num in mod_version_attr])
        return joined_tuple
    elif callable(mod_version_attr):
        return mod_version_attr()
    else:
        # print(f'Does not support module version of type {type(mod_ver_attr)}')
        return 'NA'


# Straight from https://stackoverflow.com/a/52187331/2166823
# Slight modifications added
def _notebook_basename():
    """Returns the absolute path of the Notebook or None if it cannot be determined
    NOTE: works only when the security is token-based or there is also no password
    """
    import ipykernel
    import json
    import urllib
    from notebook import notebookapp

    connection_file = Path(ipykernel.get_connection_file()).name
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]
    for srv in notebookapp.list_running_servers():
        if srv['token'] == '' and not srv['password']:  # No token and no password
            req = urllib.request.urlopen(srv['url']+'api/sessions')
        else:
            req = urllib.request.urlopen(srv['url']+'api/sessions?token='+srv['token'])
        sessions = json.load(req)
        for sess in sessions:
            if sess['kernel']['id'] == kernel_id:
                return Path(sess['notebook']['path']).stem
    return None


def sinfo(na=True, os=True, cpu=True, jupyter=None, dependencies=None,
          std_lib=False, private=False, write_req_file=None, req_file_name=None,
          html=False, excludes=['builtins', 'stdlib_list']):
    '''
    Output version information for loaded modules in the current session,
    Python, and the OS.

    Parameters
    ----------
    na : bool
        Output module name even when no version number is found.
    os : bool
        Output OS information.
    cpu : bool
        Output number of logical CPU cores and info string (if available).
    jupyter : bool
        Output information about the jupyter environment. If `None`, output
        jupyter info only if inside a Jupyter notebook.
    dependencies : bool
        Output information about modules imported by the Python interpreter on
        startup and depency modules imported via other modules. If `None`,
        dependency modules will be included in the HTML output under a
        <details> tag, and excluded from the printed output. Setting `na` to
        `False` could be helpful to reduce verboseness.
    std_lib : bool
        Output information for modules imported from the standard library.
        Tries to detect the Python version to compare with the corresponding
        standard libarary, falls back to Python 3.7 if the version cannot be
        detected.
    private : bool
        Output information for private modules.
    write_req_file: bool
        Create a pip-compatible text file that lists all the module versions.
        If `None`, write dependency files for Jupyter notebooks only. Tries to
        find the notebook name to prefix the requirments file, falls back to
        `sinfo-requirements.txt`. Only writes explicitly imported modules.
    req_file_name : str
        Change the name of the requirements file.
    html: bool
        Format output as HTML and collapse it in a <details> tag. If `None`,
        HTML will be used only if a Jupyter notebook environment is detected.
        Note that this will not be visible in notebooks shared on GitHub since
        they seemingly do not support the <details> tag. Requires IPython.
    excludes : list
        Do not output version information for these modules.
    '''
    # Exclude std lib packages
    if not std_lib:
        try:
            std_modules = stdlib_list(version=platform.python_version()[:-2])
        except ValueError:
            # Use 3.7 if the Python version cannot be found
            std_modules = stdlib_list('3.7')
        excludes = excludes + std_modules

    # Include jupyter info
    in_notebook = 'jupyter_core' in sys.modules.keys()
    if in_notebook:
        if html is None:
            html = True
    if jupyter or (jupyter is None and in_notebook):
        jupyter = True
        jup_mod_names = ['IPython', 'jupyter_client', 'jupyter_core',
                         'jupyterlab', 'notebook']
        jup_modules = []
        for jup_mod_name in jup_mod_names:
            try:
                jup_modules.append(import_module(jup_mod_name))
            except ModuleNotFoundError:
                pass
        # The length of `'jupyter_client'` is 14
        # The spaces are added to create uniform whitespace in the output
        # f-strings, which is needed to clean them with inspect.cleandoc
        jup_mod_and_ver = [f'            {module.__name__:14}\t{module.__version__}'
                           for module in jup_modules]
        output_jupyter = '\n'.join(jup_mod_and_ver)
    else:
        output_jupyter = None

    # Get `globals()` from the enviroment where the function is executed
    caller_globals = dict(
        inspect.getmembers(inspect.stack()[1][0]))["f_globals"]
    # Find imported modules in the global namespace
    imported_modules = set(_imports(caller_globals.items()))
    # Wether to include dependency module not explicitly imported
    all_modules = {'imported': imported_modules}
    if dependencies is not False:  # Default with HTML is to include deps
        if html or dependencies:
            dependencies = True  # HTML default, used later for output strings
            sys_modules = set(sys.modules.keys())
            depend_modules = sys_modules.difference(imported_modules)
            all_modules['depend'] = depend_modules

    output_modules = {}
    for mod_type in all_modules:
        modules = all_modules[mod_type]
        # Keep module basename only. Filter duplicates and excluded modules.
        if private:
            clean_modules = set(module.split('.')[0] for module in modules
                                if module.split('.')[0] not in excludes)
        else:  # Also filter private modules
            clean_modules = set(module.split('.')[0] for module in modules
                                if module.split('.')[0] not in excludes
                                and not module.startswith('_'))

        # Don't duplicate jupyter module output
        if jupyter or in_notebook:
            for jup_module in jup_modules:
                if jup_module.__name__ in clean_modules:
                    clean_modules.remove(jup_module.__name__)

        # Find version number. Return NA if a version string can't be found.
        # This section is modified from the `py_session` package
        mod_and_ver = []
        mod_names = []
        mod_versions = []
        for mod_name in clean_modules:
            mod_names.append(mod_name)
            mod = sys.modules[mod_name]
            # Since modules use different attribute names to store version info,
            # try the most common ones.
            try:
                mod_version = _find_version(mod.__version__)
            except AttributeError:
                try:
                    mod_version = _find_version(mod.version)
                except AttributeError:
                    try:
                        mod_version = _find_version(mod.VERSION)
                    except AttributeError:
                        mod_version = 'NA'
                        # print(f'Cannot find a version attribute for {mod}.')
            mod_versions.append(mod_version)
        max_name_len = max([len(mod_name) for mod_name in mod_names])
        mod_and_ver = [f'{mod_name:{max_name_len}}\t{mod_version}'
                       for mod_name, mod_version in zip(mod_names, mod_versions)]
        if not na:
            mod_and_ver = [x for x in mod_and_ver if not x[-2:] == 'NA']
        mod_and_ver = sorted(mod_and_ver)
        output_modules[mod_type] = '\n            '.join(mod_and_ver)

    # Write requirements file for notebooks only by default
    if write_req_file or (write_req_file is None and in_notebook):
        if req_file_name is None:
            # Only import notebook librarie if we are in the notebook
            # Otherwise, running this multiple times would set `in_notebook=True`
            if in_notebook:
                try:
                    req_file_name = f'{_notebook_basename()}-requirements.txt'
                except:  # Fallback if the notebook name cannot be found
                    req_file_name = 'sinfo-requirements.txt'
            else:
                req_file_name = 'sinfo-requirements.txt'

        # For NA modules, just include the latest, so no version number.
        mods_na_removed = [mod_ver.replace('\tNA', '')
                           for mod_ver in output_modules['imported'].split('\n')]
        if jupyter:
            mods_req_file = mods_na_removed + jup_mod_and_ver
        else:
            mods_req_file = mods_na_removed
        clean_mods_req_file = [mod_ver.replace(' ', '').replace('\t', '==')
                               for mod_ver in mods_req_file]
        with open(req_file_name, 'w') as f:
            for mod_to_req in clean_mods_req_file:
                f.write('{}\n'.format(mod_to_req))

    # Sys info
    sys_output = 'Python ' + sys.version.replace('\n', '')
    os_output = platform.platform() if os else None
    if cpu:
        if platform.processor() != '':
            cpu_output = f'{cpu_count()} logical CPU cores, {platform.processor()}'
        else:
            cpu_output = f'{cpu_count()} logical CPU cores'
    else:
        cpu_output = None
    date_output = 'Session information updated at {}'.format(
        datetime.now().strftime('%Y-%m-%d %H:%M'))

    # Output
    # For proper formatting and exclusion of `-----` when `jupyter=False`.
    nl = '\n'
    output_jup_str = ('' if output_jupyter is None
                      else f'{nl}            -----{nl}{output_jupyter}')
    if html:
        from IPython.display import HTML
        if dependencies:
            # Must be dedented to line up with the returned HTML.
            # Otherwise `cleandoc()` does not work.
            output_depend_str = f"""
            -----
            </pre>
            <details>
            <summary>Click to view dependency modules</summary>
            <pre>
            {output_modules['depend']}
            </pre>
            </details> <!-- seems like this ends pre, so might as well be explicit -->
            <pre>"""
        else:
            output_depend_str = ''
        return HTML(cleandoc(f"""
            <details>
            <summary>Click to view module versions</summary>
            <pre>
            -----
            {output_modules['imported']}{output_depend_str}{output_jup_str}
            -----
            {sys_output}
            {os_output}
            {cpu_output}
            -----
            {date_output}
            </pre>
            </details>"""))
    else:
        if dependencies:
            # Must be dedented to line up with the returned HTML.
            # Otherwise `cleandoc()` does not work.
            output_depend_str = f"""
            -----
            {output_modules['depend']}"""
        else:
            output_depend_str = ''
        print(cleandoc(f"""
            -----
            {output_modules['imported']}{output_depend_str}{output_jup_str}
            -----
            {sys_output}
            {os_output}
            {cpu_output}
            -----
            {date_output}"""))

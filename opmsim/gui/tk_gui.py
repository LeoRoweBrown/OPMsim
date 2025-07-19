import tkinter as tk
from tkinter import messagebox
import traceback
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from matplotlib import pyplot as plt
from inspect import signature
from typing import get_type_hints, get_args
import numpy as np

from opmsim.optical_system import OpticalSystem
from opmsim.dipole_source import DipoleSource
from opmsim.optical_elements.sine_lens import SineLens
from opmsim.optical_elements.wave_plate import WavePlate
from opmsim.optical_elements.linear_polariser import LinearPolariser
from opmsim.optical_elements.flat_mirror import FlatMirror, ProtectedFlatMirror, UncoatedFlatMirror
from opmsim.gui.tk_widgets import ElementFrame, ScrollableFrame
from opmsim.visualization.dipole_plot_for_gui import plot_dipole_source_3d
from opmsim.visualization.optical_system_diagram import OpticalSystemDiagram
# from opmsim.visualization.ray_tracing_plots import

class SystemDesignerApp():
    def __init__(self, root) -> None:
        self.root = root
        self.width = 1280
        self.height = 720
        self.root.title("OPMsim")
        self.root.geometry(f"{self.width}x{self.height}")  # Set window size directly
        # self.canvas = tk.Canvas(root, width=self.width, height=self.height)
        # self.canvas.pack()

        self.elementClasses = [
            SineLens,
            WavePlate,
            LinearPolariser,
            FlatMirror,
            ProtectedFlatMirror,
            UncoatedFlatMirror
        ]
        self.element_info = {}
        for element in self.elementClasses:
            name = element.__name__
            self.element_info[name] = {
                'class': element,
                'args': self.class_to_schema(element),
                'current_arg_vals': []
            }
        print(self.element_info)

        self.init_main_sim_bar()
        self.init_optical_system()
        self.init_element_config_bar()
        self.init_source_config_bar()
        self.init_menu_bar()

        tk.Tk.report_callback_exception = self.show_uncaught_error

    def init_main_sim_bar(self):
        self.main_sim_frame = tk.Frame(self.root, width=640, highlightbackground="red", highlightthickness=2)
        self.main_sim_frame.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        self.main_sim_frame.pack_propagate(False)

        self.ray_diagram_frame = tk.Frame(
            self.main_sim_frame, highlightbackground="green", highlightthickness=2)
        self.ray_diagram_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Init the system diagram as an empty plot
        self.system_diagram_fig, self.system_diagram_ax = plt.subplots()
        self.diagram_mpl_canvas = FigureCanvasTkAgg(self.system_diagram_fig, self.ray_diagram_frame)
        self.diagram_mpl_canvas.draw()
        self.diagram_mpl_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # self.mpl_canvas.get_tk_widget().pack_propagate(True)

        # Init the system object and diagram (empty, no elements or source)
        self.optical_system = OpticalSystem()
        self.optical_system.debug_plots = True
        self.system_diagram = OpticalSystemDiagram(self.optical_system)

        self.sim_control_frame = tk.Frame(self.main_sim_frame, highlightbackground="blue",
                                          background='green', highlightthickness=2)
        self.sim_control_frame.pack(side=tk.TOP, fill=tk.BOTH)
        self.n_rays_tk_var = tk.StringVar()
        self.n_rays_tk_var.set('200')
        self.n_rays_field = tk.Entry(self.sim_control_frame, textvariable=self.n_rays_tk_var)
        self.preview_rays_button = tk.Button(
            self.sim_control_frame, text='Preview ray-tracing', command=self.preview_raytrace)
        self.preview_rays_button.pack(side=tk.LEFT)
        self.calculate_efields_button = tk.Button(
            self.sim_control_frame, text='Run simulation', command=self.preview_raytrace)
        self.calculate_efields_button.pack(side=tk.LEFT)

    def init_source_config_bar(self):
        self.source_config_container = tk.Frame(
            self.root, width=320, highlightbackground="blue", highlightcolor='blue', highlightthickness=2)
        self.source_config_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=0, pady=0)
        self.source_config_container.pack_propagate(False)  # Prevent the frame from resizing to fit child widgets

        # 3D plot widget of dipole source GUI element
        self.dipole_view_frame = tk.Frame(self.source_config_container, height=320,
                                          highlightbackground="green", highlightthickness=5)
        self.dipole_view_label = tk.Label(self.dipole_view_frame, text="Dipole source")
        self.dipole_view_label.pack()
        self.dipole_fig, self.dipole_ax = plt.subplots()
        self.mpl_canvas = FigureCanvasTkAgg(self.dipole_fig, self.dipole_view_frame)
        self.mpl_canvas.draw()
        self.mpl_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.dipole_view_frame.pack(expand=False, fill=tk.X)
        self.dipole_view_frame.pack_propagate(False)

        # Dipole settings GUI elements
        self.configure_dipoles_frame = tk.Frame(
            self.source_config_container
        )
        self.configure_dipoles_frame.pack(fill=tk.X, padx=5, pady=5)
        self.dipole_source_label = tk.Label(self.configure_dipoles_frame, text="Dipole source type")
        self.dipole_source_label.grid(row=1, column=1)
        self.dipole_source_combobox = ttk.Combobox(self.configure_dipoles_frame, state='readonly')
        self.dipole_source_combobox['values'] = [
            'Uniform source',
            'x-dipole',
            'y-dipole',
            'z-dipole'
        ]
        self.dipole_source_combobox.current(1)
        self.dipole_source_combobox.grid(row=1, column=2, padx=2, pady=2)
        self.dipole_source_combobox.bind("<<ComboboxSelected>>", self.set_dipole_source)

        self.dipole_count = 500
        self.dipole_count_tk_var = tk.StringVar()
        self.dipole_count_tk_var.set(f'{self.dipole_count}')
        self.dipole_count_label = tk.Label(self.configure_dipoles_frame, text="Dipole count")
        self.dipole_count_label.grid(row=2, column=1, padx=2, pady=2)
        self.dipole_count_entry = tk.Entry(
            self.configure_dipoles_frame, textvariable=self.dipole_count_tk_var)
        self.dipole_count_entry.bind("<Return>", self.update_dipole_count)
        # self.dipole_count_entry.bind("<FocusOut>", self.update_dipole_count)
        self.dipole_count_entry.grid(row=2, column=2, padx=2, pady=2)

        # Excitation config GUI elements
        self.excitation_var_phi = tk.StringVar()
        self.excitation_var_alpha = tk.StringVar()
        self.excitation_label = tk.Label(self.configure_dipoles_frame, text="Excitation direction")
        self.excitation_label.grid(row=3, column=1, padx=2, pady=2)
        self.excitation_combobox = ttk.Combobox(self.configure_dipoles_frame, state='readonly')
        self.excitation_combobox['values'] = [
            'Unpolarised/rapidly tumbling',
            'x-dipole',
            'y-dipole',
            'z-dipole'
        ]
        self.excitation_combobox.current(0)
        self.excitation_combobox.grid(row=3, column=2, padx=2, pady=2)
        self.excitation_combobox.bind("<<ComboboxSelected>>", self.set_excitation)
        self.update_dipole_source_button = tk.Button(
            self.configure_dipoles_frame, text="Update", command=self.set_dipole_source)
        self.update_dipole_source_button.grid(row=4, column=2, padx=2, pady=2)

        self.set_dipole_source()

    def init_element_config_bar(self):
        self.element_list_and_config_container = tk.Frame(
            self.root, width=320, highlightbackground="blue", highlightthickness=2)
        self.element_list_and_config_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=0, pady=0)
        self.element_list_and_config_container.pack_propagate(False)  # Prevent the frame from resizing to fit child widgets

        self.init_element_list_widget(self.element_list_and_config_container)
        self.init_add_element_widget(self.element_list_and_config_container)

    def preview_raytrace(self, *args):
        self.optical_system.trace_system(preview=True)
        self.system_diagram = OpticalSystemDiagram(self.optical_system)
        self.system_diagram.draw_rays()
        self.system_diagram_fig, self.system_diagram_ax = self.system_diagram.fig, self.system_diagram.ax
        self.diagram_mpl_canvas.get_tk_widget().destroy()
        self.diagram_mpl_canvas = FigureCanvasTkAgg(self.system_diagram_fig, self.ray_diagram_frame)
        self.diagram_mpl_canvas.draw()
        self.diagram_mpl_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def set_excitation(self, *args):
        """Called by set_dipole to set photoselection"""
        selection = self.excitation_combobox.get()
        if self.dipole_source is None:
            print("no dipole source")
            return
        if selection == 'x-dipole':
            self.dipole_source.classical_photoselection((0, 0))
        elif selection == 'y-dipole':
            self.dipole_source.classical_photoselection((np.pi / 2, 0))
        elif selection == 'z-dipole':
            self.dipole_source.classical_photoselection((0, np.pi / 2))
        elif selection == 'Unpolarised/rapidly tumbling':
            # reset the photoselection
            self.dipole_source.emission_scaling = np.ones_like(self.dipole_source.emission_scaling)

    def update_dipole_count(self, *args):
        try:
            self.dipole_count = int(self.dipole_count_tk_var.get())
            self.set_dipole_source()
        except ValueError as e:
            print(e)

    def set_dipole_source(self, *args):
        selection = self.dipole_source_combobox.get()
        if selection != 'Uniform source':
            self.dipole_count_entry.config(state="disabled")
        else:
            self.dipole_count_entry.config(state="normal")

        if selection == 'x-dipole':
            self.dipole_source = DipoleSource((0, 0))
        elif selection == 'y-dipole':
            self.dipole_source = DipoleSource((np.pi / 2, 0))
        elif selection == 'z-dipole':
            self.dipole_source = DipoleSource((0, np.pi / 2))
        elif selection == 'Uniform source':
            self.dipole_source = DipoleSource()
            self.dipole_source.generate_dipole_ensemble(self.dipole_count)

        self.set_excitation()  # new dipole source is generated, need to set photoselection/excitation again
        self.optical_system.source = self.dipole_source  # update source of optical system

        # Draw 3D plot of dipole source
        alphas, phis = self.dipole_source.alpha_d, self.dipole_source.phi_d,
        scaling = self.dipole_source.emission_scaling
        self.dipole_fig = plot_dipole_source_3d(alphas, phis, scaling, show_plot=False, dipole_style='arrow')
        self.dipole_fig.set_tight_layout(True)
        self.mpl_canvas.get_tk_widget().destroy()
        self.mpl_canvas = FigureCanvasTkAgg(self.dipole_fig, self.dipole_view_frame)
        self.mpl_canvas.draw()
        self.mpl_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def init_optical_system(self):
        self.optical_system = OpticalSystem()

    def class_to_schema(self, cls):
        classname = cls.__name__
        sig = signature(cls.__init__)
        hints = get_type_hints(cls.__init__)
        # below is to handle typing.Optional[T] in the hints (primitives e.g. float are fine).
        hints = dict([(key, get_args(hint)[0]) if len(get_args(hint)) != 0 else (key, hint)
                      for key, hint in hints.items()])
        print(hints)
        schema = []  # [{'element': classname}]
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            default = param.default if param.default is not param.empty else None
            print("hint get", hints.get(name, str))
            field = {
                'name': name,
                'type': type(default) if default is not None else hints.get(name, str),
                'type_str': type(default).__name__ if default is not None else hints.get(name, str).__name__,
                'default': default,
            }
            schema.append(field)
        return schema

    def add_element(self):
        name = self.element_str_var.get()
        Element = self.element_info[name]['class']  # get class

        # get options
        options = {}
        options_str = ''
        # var is the StringVar BooleanVar from tkinter
        for n, var in enumerate(self.current_config_vars):
            kw = var['arg_name']
            val = var['arg_var'].get()
            a_type = var['arg_type']
            if val == '':
                continue
            options[kw] = (a_type)(val)
            options_str += f'{kw}: {val}, '

        try:
            element = Element(**options)
        except Exception as e:
            self.error_box(e)
            return

        self.optical_system.elements.append(element)
        # Element number Label
        element_idx = len(self.element_list_widget.element_list_entries)

        element_entry_frame = ElementFrame(self.element_list_widget.element_list_frame, element_idx,
                                           name, options_str, self.remove_element,
                                           width=300)
        self.element_list_widget.element_list_entries.append(element_entry_frame)
        element_entry_frame.pack()
        self.update_element_diagram()

    def remove_element(self, idx):
        self.optical_system.elements.pop(idx)
        self.element_list_widget.element_list_entries.pop(idx)
        # updates indices
        print("Updating index")
        for n, entry in enumerate(self.element_list_widget.element_list_entries):
            print(n, entry)
            entry.set_index(n)
            print(entry.element_idx)
            print(entry.name)
            print(self.optical_system.elements[n])
        self.update_element_diagram()

    def update_element_diagram(self):
        self.system_diagram.draw_elements()

    def show_element_config(self, event):
        for widget in self.element_config_frame.winfo_children():
            widget.destroy()
        element_str = self.drop_down_element.get()
        args = self.element_info[element_str]['args']
        # self.element_config_frame.grid(column=1, row=len(args)+1)
        title_arg = tk.Label(self.element_config_frame, text="Parameter")
        title_val = tk.Label(self.element_config_frame, text="Value")
        title_arg.grid(column=0, row=0)
        title_val.grid(column=1, row=0)
        # get the last used values the user typed in
        saved_arg_vals = self.element_info[element_str]['current_arg_vals']
        self.current_config_vars = []
        for n, arg in enumerate(args):
            arg_name = arg['name']
            arg_type = arg['type']
            print(saved_arg_vals)
            print(arg_name)
            arg_val = arg['default'] if len(saved_arg_vals) == 0 else saved_arg_vals[arg_name]
            if arg_type is bool or type(arg_val) is bool:
                var = tk.BooleanVar()
                tk_arg = tk.Checkbutton(self.element_config_frame, text=arg_name, variable=var)
                if arg_val is not None:
                    var.set(bool(arg_val))
                # tk_arg.pack()
                tk_arg.grid(column=0, row=(n + 1))
            else:
                var = tk.StringVar()
                tk_label = tk.Label(self.element_config_frame, text=f'{arg_name} ({arg_type.__name__})')
                tk_arg = tk.Entry(self.element_config_frame, textvariable=var)  # , width=50)
                if arg_val is not None:
                    var.set(str(arg_val))
                # tk_label.pack()
                # tk_arg.pack(expand=True, fill=tk.X)
                tk_label.grid(column=0, row=(n + 1))
                tk_arg.grid(column=1, row=(n + 1))
            self.current_config_vars.append({'arg_name': arg_name, 'arg_type': arg_type, 'arg_var': var})
            var.trace_add('write', self.update_element_options)  # add trace so values get updated in element_class_info

    def init_menu_bar(self):
        self.menubar = tk.Menu()
        filemenu = tk.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="New", command=self.donothing)
        filemenu.add_command(label="Open", command=self.donothing)
        filemenu.add_command(label="Save", command=self.donothing)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=filemenu)

    def donothing(self):
        pass

    def init_element_list_widget(self, parent_container):
        self.element_list_widget = ScrollableFrame(
            parent_container, width=300, highlightbackground="gray", highlightthickness=2)

    def init_add_element_widget(self, parent_continer):
        self.element_widget_frame = tk.Frame(parent_continer, height=100,
                                             highlightbackground="gray", highlightthickness=2)
        self.element_widget_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=10, pady=10)
        self.element_str_var = tk.StringVar()
        self.drop_down_element = ttk.Combobox(self.element_widget_frame, width=15, state='readonly',
                                              textvariable=self.element_str_var)
        self.drop_down_element['values'] = list(self.element_info.keys())

        self.drop_down_element.bind("<ButtonRelease-1>", self.update_element_options)
        self.drop_down_element.bind("<<ComboboxSelected>>", self.show_element_config)
        # self.drop_down_element.grid(column=1, row=len(self.drop_down_element['values']))
        self.drop_down_element.grid(column=0, row=0)  # pack(side=tk.TOP, padx=5)
        self.drop_down_element.current(0)
        self.drop_down_element.get()
        self.add_element_button = tk.Button(self.element_widget_frame,
                                            text="Add",
                                            command=lambda: self.add_element())
        self.element_config_frame = tk.Frame(self.element_widget_frame,
                                             highlightbackground="gray", highlightthickness=2)
        self.element_config_frame.grid(column=0, row=1)  # pack()
        self.add_element_button.grid(column=0, row=2)  # pack(side=tk.BOTTOM, padx=5)
        self.show_element_config(tk.Event())

    def get_current_element_args(self):
        element_str = self.drop_down_element.get()
        args = self.element_info[element_str]['args']
        options = {}
        options_str = ''
        for n, var in enumerate(self.current_config_vars):
            kw = var['arg_name']
            val = var['arg_var'].get()
            a_type = var['arg_type']
            print(a_type, val)
            options[kw] = (val)  # don't need to cast here, casting handled in add_element
            options_str += f'{kw}: {val}, '
        return element_str, options, options_str

    def update_element_options(self, *args):
        element_str, options, options_str = self.get_current_element_args()
        self.element_info[element_str]['current_arg_vals'] = options

    def show_uncaught_error(self, *args):
        err = traceback.format_exception(*args)
        messagebox.showerror('Exception', err)
        print(err)

    def error_box(self, err):
        messagebox.showerror("Error", err)
        print(err)

def main():
    root = tk.Tk()
    app = SystemDesignerApp(root)
    root.config(menu=app.menubar)
    root.mainloop()

if __name__ == "__main__":
    main()

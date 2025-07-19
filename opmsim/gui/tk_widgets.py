import tkinter as tk

class ElementFrame(tk.Frame):
    def __init__(self, parent, index, name, options_str='', on_remove=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.element_idx = index
        self.name = name
        self.on_remove = on_remove  # callback function that gets called on removal of self
        # Element number Label
        self.idx_label = tk.StringVar()
        self.idx_label.set(f'{index}')
        self.element_num_label = tk.Label(self, textvariable=self.idx_label)
        self.element_num_label.pack(side=tk.LEFT)

        # Text widget with controlled width
        self.element_text_widget = tk.Text(self, height=3, width=30, wrap='word')
        self.element_text_widget.insert('1.0', f'{name}: {options_str}')
        self.element_text_widget.pack(side=tk.LEFT)

        # Add the remove button
        self.remove_element_button = tk.Button(self, text="-", width=1,
                                               command=self.remove)
        self.remove_element_button.pack(side=tk.RIGHT)

    def remove(self):
        if self.on_remove:
            self.on_remove(self.element_idx)
        self.destroy()

    def set_index(self, index):
        self.element_idx = index
        self.idx_label.set(f'{index}')

class ScrollableFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        # Create container, frame canvas that can scroll, and frame containing the list
        self.element_list_canvas = tk.Canvas(
            self, width=280, highlightbackground="gray", highlightthickness=2)
        width = self.winfo_reqwidth()  # double check this
        self.element_list_scrollbar = tk.Scrollbar(self, width=20)
        self.element_list_frame = tk.Frame(
            self.element_list_canvas, width=(width - 20), highlightbackground="gray", highlightthickness=2)
        element_list_title = tk.Label(self, text="Element list")

        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=0, pady=0)
        element_list_title.pack(side=tk.TOP, fill=tk.X, expand=False)
        self.element_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.element_list_canvas.pack_propagate(False)

        # dont pack the list frame, use window instead to view it
        self.element_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))

        self.element_list_scrollbar.config(command=self.element_list_canvas.yview)
        self.element_list_canvas.configure(yscrollcommand=self.element_list_scrollbar.set)

        # Update the scroll region every time element is added/removed
        self.element_list_frame.bind("<Configure>", self.update_list_bbox)

        self.element_list_canvas.create_window(
            (0, 0), window=self.element_list_frame, anchor="nw")

        self.element_list_entries = []

    def update_list_bbox(self, e):
        self.element_list_canvas.configure(
            scrollregion=self.element_list_canvas.bbox("all"))
        print(self.element_list_canvas.bbox("all"))


class PolarAngleEntryWidget(tk.Frame):
    def __init__(self, container):
        """
        Reusable little widget for phi and alpha entry boxes. Not currently used
        Args:
            container (Tk.Frame): container frame

        Returns:
            tuple: polar_angle_frame,
                (tk.Entry for alpha, tk.StringVar for alpha),
                (tk.Entry for phi, tk.StringVar for phi)
        """
        # Frame containing the phi, alpha entry
        polar_angle_frame = tk.Frame(container)
        var_phi = tk.StringVar()
        var_alpha = tk.StringVar()
        label_phi = tk.Label(polar_angle_frame, text="phi")
        label_phi.pack(side=tk.LEFT)
        phi_entry = tk.Entry(
            polar_angle_frame, textvariable=var_phi, width=7
        )
        phi_entry.pack(side=tk.LEFT)
        label_alpha = tk.Label(polar_angle_frame, text="alpha")
        label_alpha.pack(side=tk.LEFT)
        alpha_entry = tk.Entry(
            polar_angle_frame, textvariable=var_alpha, width=7
        )
        alpha_entry.pack(side=tk.LEFT)
        return polar_angle_frame, (alpha_entry, var_alpha), (phi_entry, var_phi)

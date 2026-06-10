import csv
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


def initialize_lattice(L, start="up", rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if start == "up":
        return np.ones((L, L), dtype=int)
    if start == "down":
        return -np.ones((L, L), dtype=int)
    return rng.choice([-1, 1], size=(L, L))


def total_energy(spins, J=1.0, h=0.0):
    right_neighbors = np.roll(spins, -1, axis=1)
    down_neighbors = np.roll(spins, -1, axis=0)
    return float(-J * np.sum(spins * (right_neighbors + down_neighbors)) - h * np.sum(spins))


def total_magnetization(spins):
    return int(np.sum(spins))


def metropolis_sweep(spins, temperature, rng, J=1.0, h=0.0):
    L = spins.shape[0]
    n_sites = L * L
    accepted = 0
    delta_e_total = 0.0
    delta_m_total = 0

    for _ in range(n_sites):
        i = rng.integers(0, L)
        j = rng.integers(0, L)
        s = spins[i, j]
        neighbor_sum = (
            spins[(i + 1) % L, j]
            + spins[(i - 1) % L, j]
            + spins[i, (j + 1) % L]
            + spins[i, (j - 1) % L]
        )
        delta_E = 2.0 * s * (J * neighbor_sum + h)
        if delta_E <= 0.0 or (temperature > 0.0 and rng.random() < np.exp(-delta_E / temperature)):
            spins[i, j] = -s
            accepted += 1
            delta_e_total += delta_E
            delta_m_total += -2 * s

    return accepted / n_sites, delta_e_total, delta_m_total


def spin_correlation_profile(spins):
    L = spins.shape[0]
    max_r = L // 2
    profile = np.empty(max_r + 1, dtype=float)
    spins_float = spins.astype(float, copy=False)

    for r in range(max_r + 1):
        corr_x = np.mean(spins_float * np.roll(spins_float, -r, axis=1))
        corr_y = np.mean(spins_float * np.roll(spins_float, -r, axis=0))
        profile[r] = 0.5 * (corr_x + corr_y)

    return profile


class IsingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Ising Model Desktop GUI")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 750)
        self.root.withdraw()

        self.exact_tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
        self.spin_cmap = ListedColormap(["#1d3557", "#e9c46a"])
        self.colors = {
            "bg": "#d9d9d9",
            "panel": "#d9d9d9",
            "panel_alt": "#d9d9d9",
            "border": "#a9a9a9",
            "text": "#000000",
            "muted": "#000000",
            "accent": "#d9d9d9",
            "accent_dark": "#c6c6c6",
            "accent_soft": "#efefef",
            "success": "#2f6f4f",
            "focus": "#d9d9d9",
            "tab_active": "#ececec",
            "tab_idle": "#d9d9d9",
            "tab_selected": "#f2f2f2",
        }
        self.fonts = {
            "title": ("TkDefaultFont", 10, "bold"),
            "section": ("TkDefaultFont", 10, "bold"),
            "body": ("TkDefaultFont", 10),
            "small": ("TkDefaultFont", 9),
            "mono": ("TkFixedFont", 10),
            "control_body": ("TkDefaultFont", 10),
            "control_small": ("TkDefaultFont", 9),
            "control_mono": ("TkFixedFont", 10),
            "tab": ("TkDefaultFont", 10),
        }
        self.root.configure(bg=self.colors["bg"])
        self.style = ttk.Style()
        self._configure_theme()

        self.rng = np.random.default_rng(12345)
        self.run_in_progress = False
        self.run_paused = False
        self.run_state = None

        self.spins = None
        self.current_temperature = None
        self.time_energy = []
        self.time_magnetization = []
        self.time_acceptance = []
        self.scan_rows = []
        self.scan_correlations = []

        self._build_variables()
        self._build_layout()
        self.reset_simulation()
        self.root.after(100, self._show_window)

    def _show_window(self):
        self.root.update_idletasks()

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        width = min(1400, max(1100, screen_w - 120))
        height = min(900, max(750, screen_h - 120))
        x = max(40, (screen_w - width) // 2)
        y = max(40, (screen_h - height) // 2)

        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.state("normal")
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(250, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()
        self.root.after(300, lambda: self._ensure_visible(3))

    def _ensure_visible(self, retries):
        if retries <= 0:
            return

        try:
            self.root.state("normal")
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.root.after(250, lambda: self._ensure_visible(retries - 1))
        except tk.TclError:
            return

    def _build_variables(self):
        self.lattice_size_var = tk.IntVar(value=10)
        self.j_var = tk.DoubleVar(value=1.0)
        self.h_var = tk.DoubleVar(value=0.0)
        self.seed_var = tk.IntVar(value=12345)
        self.replica_count_var = tk.IntVar(value=4)
        self.start_mode_var = tk.StringVar(value="up")
        self.equil_sweeps_var = tk.IntVar(value=600)
        self.measurement_sweeps_var = tk.IntVar(value=1000)
        self.t_min_var = tk.DoubleVar(value=2.0)
        self.t_max_var = tk.DoubleVar(value=4.0)
        self.t_step_var = tk.DoubleVar(value=0.25)
        self.refine_near_tc_var = tk.BooleanVar(value=False)
        self.refine_span_var = tk.DoubleVar(value=0.2)
        self.refine_step_var = tk.DoubleVar(value=0.05)
        self.status_var = tk.StringVar(value="Ready.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.scan_progress_var = tk.StringVar(value="Scan progress: 0/0 temperatures completed")

    def _configure_theme(self):
        self.style.theme_use("default")
        self.style.configure(
            ".",
            background=self.colors["bg"],
            foreground=self.colors["text"],
            fieldbackground=self.colors["panel"],
            font=self.fonts["body"],
        )
        self.style.configure("App.TFrame", background=self.colors["bg"])
        self.style.configure("Panel.TFrame", background=self.colors["panel"])
        self.style.configure("Info.TLabel", background=self.colors["panel"], foreground=self.colors["text"])
        self.style.configure(
            "TEntry",
            padding=3,
            fieldbackground=self.colors["panel"],
            insertcolor=self.colors["text"],
        )
        self.style.configure(
            "Visual.TNotebook",
            background=self.colors["bg"],
            borderwidth=0,
            tabmargins=(2, 2, 2, 0),
        )
        self.style.configure(
            "Visual.TNotebook.Tab",
            background=self.colors["tab_idle"],
            foreground=self.colors["text"],
            padding=(8, 4),
            font=self.fonts["tab"],
        )
        self.style.map(
            "Visual.TNotebook.Tab",
            background=[
                ("selected", self.colors["tab_selected"]),
                ("active", self.colors["tab_active"]),
            ],
            foreground=[("selected", self.colors["text"]), ("active", self.colors["text"])],
        )
        self.style.configure(
            "Vertical.TScrollbar",
            troughcolor=self.colors["panel"],
            background=self.colors["panel"],
            arrowcolor=self.colors["text"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["panel"],
            darkcolor=self.colors["panel"],
            gripcount=0,
        )

    def _build_layout(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls_host = tk.Frame(self.root, bg=self.colors["bg"], width=335)
        controls_host.grid(row=0, column=0, sticky="ns")
        controls_host.grid_propagate(False)
        controls_host.rowconfigure(0, weight=1)
        controls_host.columnconfigure(0, weight=1)

        self.controls_canvas = tk.Canvas(
            controls_host,
            width=335,
            highlightthickness=0,
            bd=0,
            bg=self.colors["bg"],
        )
        self.controls_canvas.grid(row=0, column=0, sticky="nsew")

        controls_scrollbar = ttk.Scrollbar(
            controls_host,
            orient="vertical",
            command=self.controls_canvas.yview,
            style="Vertical.TScrollbar",
        )
        controls_scrollbar.grid(row=0, column=1, sticky="ns")
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        controls = tk.Frame(self.controls_canvas, bg=self.colors["bg"])
        self.controls_window = self.controls_canvas.create_window((0, 0), window=controls, anchor="nw")

        controls.bind("<Configure>", self._on_controls_configure)
        self.controls_canvas.bind("<Configure>", self._on_controls_canvas_configure)
        self.controls_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        right = ttk.Frame(self.root, style="App.TFrame", padding=(6, 6, 6, 6))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        controls.columnconfigure(0, weight=1)

        tk.Label(
            controls,
            text="Controls",
            bg=self.colors["bg"],
            fg=self.colors["text"],
            font=self.fonts["section"],
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 10))

        form_frame = tk.Frame(controls, bg=self.colors["bg"])
        form_frame.grid(row=1, column=0, sticky="ew", padx=8)
        form_frame.columnconfigure(1, weight=1)

        fields = [
            ("Lattice size L", self.lattice_size_var),
            ("Coupling J", self.j_var),
            ("Field h", self.h_var),
            ("Equilibration sweeps", self.equil_sweeps_var),
            ("Measurement sweeps", self.measurement_sweeps_var),
            ("T min", self.t_min_var),
            ("T max", self.t_max_var),
            ("T step", self.t_step_var),
            ("Replica count", self.replica_count_var),
        ]
        for row, (label, var) in enumerate(fields):
            tk.Label(form_frame, text=label, bg=self.colors["bg"], fg=self.colors["text"], anchor="w").grid(
                row=row, column=0, sticky="w", pady=2
            )
            tk.Entry(form_frame, textvariable=var).grid(row=row, column=1, sticky="ew", pady=2, padx=(6, 0))

        tk.Label(form_frame, text="Initial state", bg=self.colors["bg"], fg=self.colors["text"], anchor="w").grid(
            row=len(fields), column=0, sticky="w", pady=2
        )
        ttk.Combobox(
            form_frame,
            textvariable=self.start_mode_var,
            values=("up", "down", "random"),
            state="readonly",
        ).grid(row=len(fields), column=1, sticky="ew", pady=2, padx=(6, 0))

        button_specs = [
            ("Run", self.run_action),
            ("Pause", self.pause_action),
            ("Stop", self.stop_run),
            ("Reset", self.reset_simulation),
            ("Export CSV", self.export_scan_csv),
        ]
        self.control_buttons = []
        buttons_frame = tk.Frame(controls, bg=self.colors["bg"])
        buttons_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(12, 8))
        for index in range(3):
            buttons_frame.columnconfigure(index, weight=1)
        for index, (label, command) in enumerate(button_specs):
            button = tk.Button(buttons_frame, text=label, command=command)
            row = index // 3
            column = index % 3
            colspan = 1
            if label == "Export CSV":
                row = 1
                column = 1
                colspan = 2
            button.grid(row=row, column=column, columnspan=colspan, sticky="ew", padx=4, pady=3, ipady=4)
            self.control_buttons.append(button)

        tk.Frame(controls, bg=self.colors["bg"], height=12, highlightbackground=self.colors["border"], highlightthickness=1).grid(
            row=3, column=0, sticky="ew", padx=8, pady=(4, 10)
        )

        stats_frame = tk.Frame(controls, bg=self.colors["bg"], highlightbackground=self.colors["border"], highlightthickness=1)
        stats_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 10))
        tk.Label(
            stats_frame,
            text="Summary",
            bg=self.colors["bg"],
            fg=self.colors["text"],
            font=self.fonts["section"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 6))
        self.summary_label = tk.Label(
            stats_frame,
            justify="left",
            anchor="nw",
            bg=self.colors["bg"],
            fg=self.colors["text"],
            font=self.fonts["body"],
        )
        self.summary_label.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))

        values_frame = tk.Frame(controls, bg=self.colors["bg"], highlightbackground=self.colors["border"], highlightthickness=1)
        values_frame.grid(row=5, column=0, sticky="ew", padx=8, pady=(0, 10))
        tk.Label(
            values_frame,
            text="Current Values",
            bg=self.colors["bg"],
            fg=self.colors["text"],
            font=self.fonts["section"],
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 8))
        self.values_label = tk.Label(
            values_frame,
            justify="left",
            anchor="nw",
            bg=self.colors["bg"],
            fg=self.colors["text"],
            font=self.fonts["body"],
        )
        self.values_label.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))

        self.status_label = tk.Label(
            controls,
            textvariable=self.status_var,
            justify="left",
            anchor="w",
            bg=self.colors["bg"],
            fg=self.colors["text"],
            wraplength=300,
        )
        self.status_label.grid(row=6, column=0, sticky="ew", padx=8, pady=(0, 12))

        notebook = ttk.Notebook(right, style="Visual.TNotebook")
        notebook.grid(row=0, column=0, sticky="nsew")

        live_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=6)
        temp_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=6)
        live_tab.columnconfigure(0, weight=1)
        live_tab.rowconfigure(0, weight=1)
        temp_tab.columnconfigure(0, weight=1)
        temp_tab.rowconfigure(0, weight=1)
        notebook.add(live_tab, text="Live Graphs")
        notebook.add(temp_tab, text="Temperature Graphs")

        self.live_figure = Figure(figsize=(8.8, 6.8), dpi=100, constrained_layout=True)
        live_gs = self.live_figure.add_gridspec(2, 2)
        self.ax_lattice = self.live_figure.add_subplot(live_gs[0, 0])
        self.ax_m_time = self.live_figure.add_subplot(live_gs[0, 1])
        self.ax_e_time = self.live_figure.add_subplot(live_gs[1, 0])
        self.ax_acc_time = self.live_figure.add_subplot(live_gs[1, 1])
        self.live_canvas = FigureCanvasTkAgg(self.live_figure, master=live_tab)
        self.live_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.temp_figure = Figure(figsize=(8.8, 6.8), dpi=100, constrained_layout=True)
        temp_gs = self.temp_figure.add_gridspec(2, 2)
        self.ax_m_temp = self.temp_figure.add_subplot(temp_gs[0, 0])
        self.ax_energy_temp = self.temp_figure.add_subplot(temp_gs[0, 1])
        self.ax_cv_temp = self.temp_figure.add_subplot(temp_gs[1, 0])
        self.ax_chi_temp = self.temp_figure.add_subplot(temp_gs[1, 1])
        self.temp_canvas = FigureCanvasTkAgg(self.temp_figure, master=temp_tab)
        self.temp_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        notebook.select(live_tab)

    def _on_controls_configure(self, _event):
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _on_controls_canvas_configure(self, event):
        self.controls_canvas.itemconfigure(self.controls_window, width=event.width)

    def _on_mousewheel(self, event):
        self.controls_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def initialize_action(self):
        self.reset_simulation()

    def run_action(self):
        if not self.run_in_progress:
            self.start_run()

    def pause_action(self):
        if self.run_in_progress and not self.run_paused:
            self.run_paused = True
            self.status_var.set("Run paused.")
            self._update_button_states()
            self._update_summary()

    def resume_action(self):
        if self.run_in_progress and self.run_paused:
            self.run_paused = False
            self.status_var.set("Resuming run...")
            self._update_button_states()
            self._update_summary()
            self.root.after(1, self._run_step)

    def step_action(self):
        if self.run_in_progress and self.run_paused:
            try:
                self._run_step_impl()
            except Exception as exc:
                self._finish_run(str(exc))

    def save_lattice_snapshot(self):
        path = filedialog.asksaveasfilename(
            title="Save lattice snapshot",
            defaultextension=".png",
            initialfile="lattice_snapshot.png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        self.ax_lattice.figure.savefig(path, dpi=150)
        self.status_var.set(f"Saved lattice snapshot to {path}")
        self._update_summary()

    def save_all_plots(self):
        path = filedialog.asksaveasfilename(
            title="Save plots",
            defaultextension=".png",
            initialfile="ising_plots.png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        self.temp_figure.savefig(path, dpi=150)
        self.status_var.set(f"Saved plots to {path}")
        self._update_summary()

    def save_scan_plots(self):
        path = filedialog.asksaveasfilename(
            title="Save scan plots",
            defaultextension=".png",
            initialfile="temperature_scan_plots.png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        self.temp_figure.savefig(path, dpi=150)
        self.status_var.set(f"Saved scan plots to {path}")
        self._update_summary()

    def export_full_sweep_csv(self):
        self.export_scan_csv()

    def temperature_grid(self):
        t_min = self.t_min_var.get()
        t_max = self.t_max_var.get()
        if t_max < t_min:
            raise ValueError("T max must be larger than or equal to T min.")
        t_step = self.t_step_var.get()
        if t_step <= 0.0:
            raise ValueError("T step must be positive.")

        temps = list(np.arange(t_min, t_max + 0.5 * t_step, t_step))
        if self.refine_near_tc_var.get():
            refine_span = self.refine_span_var.get()
            refine_step = self.refine_step_var.get()
            if refine_span < 0.0:
                raise ValueError("Refinement span must be non-negative.")
            if refine_step <= 0.0:
                raise ValueError("Refinement step must be positive.")
            refine = np.arange(
                self.exact_tc - refine_span,
                self.exact_tc + refine_span + 0.5 * refine_step,
                refine_step,
            )
            temps.extend(refine)
        return sorted({round(float(t), 10) for t in temps if t_min <= t <= t_max})

    def reset_simulation(self):
        if self.run_in_progress:
            return

        L = self.lattice_size_var.get()
        if L <= 1:
            messagebox.showerror("Invalid Lattice Size", "Lattice size must be greater than 1.")
            return

        self.rng = np.random.default_rng(self.seed_var.get())
        self.spins = initialize_lattice(L, start=self.start_mode_var.get(), rng=self.rng)
        self.current_temperature = None
        self.time_energy = [total_energy(self.spins, self.j_var.get(), self.h_var.get()) / self.spins.size]
        self.time_magnetization = [total_magnetization(self.spins) / self.spins.size]
        self.time_acceptance = [0.0]
        self.scan_rows = []
        self.scan_correlations = []
        self.run_paused = False
        self.progress_var.set(0.0)
        self.status_var.set("Simulation reset.")
        self._update_button_states()
        self._update_summary()
        self._refresh_plots()

    def _update_summary(self):
        current_e = self.time_energy[-1] if self.time_energy else 0.0
        current_m = self.time_magnetization[-1] if self.time_magnetization else 0.0
        current_acc = self.time_acceptance[-1] if self.time_acceptance else 0.0

        summary_lines = [
            f"Current T: {self.current_temperature:.3f}" if self.current_temperature is not None else "Current T: -",
            f"Current E/N: {current_e:.4f}",
            f"Current M/N: {current_m:.4f}",
            f"Acceptance ratio: {current_acc:.4f}",
            f"Tc: {self.exact_tc:.6f}",
        ]
        self.summary_label.config(text="\n".join(summary_lines))

        value_lines = [
            f"L = {self.lattice_size_var.get()}",
            f"J = {self.j_var.get():.3f}",
            f"h = {self.h_var.get():.3f}",
            f"Equil sweeps = {self.equil_sweeps_var.get()}",
            f"Meas sweeps = {self.measurement_sweeps_var.get()}",
            f"T range = {self.t_min_var.get():.3f} to {self.t_max_var.get():.3f}",
            f"T step = {self.t_step_var.get():.3f}",
            f"Initial state = {self.start_mode_var.get()}",
        ]
        if self.current_temperature is not None:
            value_lines.extend(
                [
                    f"Running T = {self.current_temperature:.3f}",
                    f"E/N = {current_e:.4f}",
                    f"M/N = {current_m:.4f}",
                    f"Acceptance = {current_acc:.4f}",
                ]
            )
        self.values_label.config(text="\n".join(value_lines))

    def _update_button_states(self):
        running = self.run_in_progress
        paused = self.run_paused
        button_states = {
            "Run": "normal" if not running else "disabled",
            "Pause": "normal" if running and not paused else "disabled",
            "Stop": "normal" if running else "disabled",
            "Reset": "normal" if not running else "disabled",
            "Export CSV": "normal" if self.scan_rows else "disabled",
        }
        for button in self.control_buttons:
            button.config(state=button_states.get(button.cget("text"), "normal"))

    def _style_axis(self, axis):
        axis.set_facecolor("#ffffff")
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_color("#000000")
        axis.tick_params(colors="#000000", labelsize=9)
        axis.title.set_color(self.colors["text"])
        axis.xaxis.label.set_color("#000000")
        axis.yaxis.label.set_color("#000000")

    def _refresh_plots(self):
        self._refresh_live_plots()
        self._refresh_temperature_plots()

    def _refresh_live_plots(self):
        x = np.arange(len(self.time_magnetization))

        self.ax_lattice.clear()
        self.ax_lattice.imshow(self.spins, cmap=self.spin_cmap, vmin=-1, vmax=1, interpolation="nearest")
        self.ax_lattice.set_title("Spin Configuration")
        self.ax_lattice.set_xticks([])
        self.ax_lattice.set_yticks([])
        self.ax_lattice.set_facecolor("#ffffff")
        self.ax_lattice.set_aspect("equal")
        for spine in self.ax_lattice.spines.values():
            spine.set_color("#000000")

        self.ax_m_time.clear()
        self.ax_m_time.plot(x, self.time_magnetization, color="red", linewidth=1.5)
        self.ax_m_time.set_title("Magnetization Per Spin vs Time")
        self.ax_m_time.set_xlabel("Monte Carlo sweeps")
        self.ax_m_time.set_ylabel("M/N")
        self._style_axis(self.ax_m_time)

        self.ax_e_time.clear()
        self.ax_e_time.plot(np.arange(len(self.time_energy)), self.time_energy, color="#26425a", linewidth=1.5)
        self.ax_e_time.set_title("Energy Per Spin vs Time")
        self.ax_e_time.set_xlabel("Monte Carlo sweeps")
        self.ax_e_time.set_ylabel("E/N")
        self._style_axis(self.ax_e_time)

        self.ax_acc_time.clear()
        self.ax_acc_time.plot(
            np.arange(len(self.time_acceptance)),
            self.time_acceptance,
            color=self.colors["success"],
            linewidth=1.5,
        )
        self.ax_acc_time.set_title("Acceptance Ratio vs Time")
        self.ax_acc_time.set_xlabel("Monte Carlo sweeps")
        self.ax_acc_time.set_ylabel("Acceptance")
        self.ax_acc_time.set_ylim(0.0, 1.05)
        self._style_axis(self.ax_acc_time)

        self.live_figure.patch.set_facecolor(self.colors["panel"])
        self.live_canvas.draw()

    def _refresh_temperature_plots(self):
        self.ax_m_temp.clear()
        self.ax_energy_temp.clear()
        self.ax_cv_temp.clear()
        self.ax_chi_temp.clear()

        if self.scan_rows:
            temps = [row["temperature"] for row in self.scan_rows]
            avg_abs_m = [row["avg_abs_magnetization_per_spin"] for row in self.scan_rows]
            avg_e = [row["avg_energy_per_spin"] for row in self.scan_rows]
            cv = [row["heat_capacity"] for row in self.scan_rows]
            chi = [row["susceptibility"] for row in self.scan_rows]

            self.ax_m_temp.plot(temps, avg_abs_m, marker="o", color="#c05050", linewidth=1.5, markersize=4)
            self.ax_energy_temp.plot(temps, avg_e, marker="o", color="#5a78a5", linewidth=1.5, markersize=4)
            self.ax_cv_temp.plot(temps, cv, marker="o", color="#c27b33", linewidth=1.5, markersize=4)
            self.ax_chi_temp.plot(temps, chi, marker="o", color="#4f4f6f", linewidth=1.5, markersize=4)

        for axis in [self.ax_m_temp, self.ax_energy_temp, self.ax_cv_temp, self.ax_chi_temp]:
            axis.set_xlabel("Temperature")
            self._style_axis(axis)
            axis.tick_params(axis="both", labelsize=9)
            axis.xaxis.set_major_locator(MultipleLocator(0.50))
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            axis.axvline(self.exact_tc, color="#9a9a9a", linestyle="--", linewidth=1.0)
            axis.set_xlim(0.5, 4.0)
            if not self.scan_rows:
                axis.set_ylim(0.0, 1.0)

        self.ax_m_temp.set_title("Magnetization vs Temperature")
        self.ax_m_temp.set_ylabel("<|M|>/N")

        self.ax_energy_temp.set_title("Energy Per Spin vs Temperature")
        self.ax_energy_temp.set_ylabel("E/N")

        self.ax_cv_temp.set_title("Specific Heat Per Spin vs Temperature")
        self.ax_cv_temp.set_ylabel("Cv")

        self.ax_chi_temp.set_title("Susceptibility vs Temperature")
        self.ax_chi_temp.set_ylabel("Chi")

        self.temp_figure.patch.set_facecolor(self.colors["panel"])
        self.temp_canvas.draw()

    def _selected_correlation_entries(self):
        if not self.scan_correlations:
            return []

        count = min(5, len(self.scan_correlations))
        if count == 1:
            return [self.scan_correlations[0]]

        indices = np.linspace(0, len(self.scan_correlations) - 1, count)
        picked = []
        seen = set()
        for raw_index in indices:
            index = int(round(raw_index))
            if index in seen:
                continue
            picked.append(self.scan_correlations[index])
            seen.add(index)

        return picked

    def start_run(self):
        if self.run_in_progress:
            return

        try:
            temperatures = self.temperature_grid()
            L = self.lattice_size_var.get()
            seed = self.seed_var.get()
            replica_count = self.replica_count_var.get()
            equil_sweeps = self.equil_sweeps_var.get()
            measurement_sweeps = self.measurement_sweeps_var.get()
        except Exception as exc:
            messagebox.showerror("Invalid Inputs", str(exc))
            return

        if L <= 1:
            messagebox.showerror("Invalid Inputs", "Lattice size must be greater than 1.")
            return
        if replica_count <= 0:
            messagebox.showerror("Invalid Inputs", "Replica count must be at least 1.")
            return
        if equil_sweeps < 0 or measurement_sweeps <= 0:
            messagebox.showerror("Invalid Inputs", "Sweep counts must be valid positive values.")
            return

        self.rng = np.random.default_rng(seed)
        self.run_in_progress = True
        self.run_paused = False
        self.progress_var.set(0.0)
        self.status_var.set("Starting run...")
        self.scan_rows = []
        self.scan_correlations = []
        self._update_button_states()
        self.run_state = {
            "temperatures": temperatures,
            "temperature_index": 0,
            "L": L,
            "n_sites": L * L,
            "J": self.j_var.get(),
            "h": self.h_var.get(),
            "seed": seed,
            "replica_count": replica_count,
            "equil_sweeps": equil_sweeps,
            "measurement_sweeps": measurement_sweeps,
            "start_mode": self.start_mode_var.get(),
            "phase": "setup",
            "phase_step": 0,
            "replica_index": 0,
            "local_equil": 0,
            "local_meas": 0,
            "spins": None,
            "energy_raw": 0.0,
            "magnetization_raw": 0.0,
            "track_correlation": False,
            "energy_sum_total": 0.0,
            "energy_sq_sum_total": 0.0,
            "magnet_sum_total": 0.0,
            "magnet_abs_sum_total": 0.0,
            "magnet_sq_sum_total": 0.0,
            "measurement_count_total": 0,
            "correlation_sum_total": None,
        }
        self._update_summary()
        self._refresh_plots()
        self.root.after(1, self._run_step)

    def toggle_pause(self):
        if not self.run_in_progress:
            return

        self.run_paused = not self.run_paused
        if self.run_paused:
            self.status_var.set("Run paused.")
            self._update_button_states()
            self._update_summary()
            return

        self.status_var.set("Resuming run...")
        self._update_button_states()
        self._update_summary()
        self.root.after(1, self._run_step)

    def stop_run(self):
        if not self.run_in_progress:
            return
        self._finish_run("stopped")

    def _run_step(self):
        try:
            self._run_step_impl()
        except Exception as exc:
            self._finish_run(str(exc))

    def _run_step_impl(self):
        if not self.run_in_progress or self.run_state is None:
            return
        if self.run_paused:
            return

        state = self.run_state
        temperatures = state["temperatures"]
        total_temperatures = len(temperatures)
        idx = state["temperature_index"]

        if idx >= total_temperatures:
            self._finish_run(None)
            return

        temperature = temperatures[idx]

        if state["phase"] == "setup":
            state["local_equil"] = max(0, state["equil_sweeps"])
            state["local_meas"] = max(1, state["measurement_sweeps"])
            state["phase_step"] = 0
            state["phase"] = "equil"
            state["replica_index"] = 0
            state["energy_sum_total"] = 0.0
            state["energy_sq_sum_total"] = 0.0
            state["magnet_sum_total"] = 0.0
            state["magnet_abs_sum_total"] = 0.0
            state["magnet_sq_sum_total"] = 0.0
            state["measurement_count_total"] = 0
            low_target = temperatures[int(np.argmin(np.abs(np.asarray(temperatures) - 1.5)))]
            near_target = temperatures[int(np.argmin(np.abs(np.asarray(temperatures) - self.exact_tc)))]
            high_target = temperatures[int(np.argmin(np.abs(np.asarray(temperatures) - 3.5)))]
            state["track_correlation"] = bool(
                np.isclose(temperature, low_target)
                or np.isclose(temperature, near_target)
                or np.isclose(temperature, high_target)
            )
            state["correlation_sum_total"] = (
                np.zeros(state["L"] // 2 + 1, dtype=float) if state["track_correlation"] else None
            )
            replica_rng = np.random.default_rng(self.rng.integers(0, 2**63 - 1))
            state["spins"] = initialize_lattice(
                state["L"], start=state["start_mode"], rng=replica_rng
            )
            state["replica_rng"] = replica_rng
            self.spins = state["spins"]
            self.current_temperature = temperature
            state["energy_raw"] = total_energy(state["spins"], J=state["J"], h=state["h"])
            state["magnetization_raw"] = float(total_magnetization(state["spins"]))
            self.time_energy = [state["energy_raw"] / state["n_sites"]]
            self.time_magnetization = [state["magnetization_raw"] / state["n_sites"]]
            self.time_acceptance = [0.0]

        batch_size = 20
        if state["phase"] == "equil":
            remaining = state["local_equil"] - state["phase_step"]
            steps = min(batch_size, remaining)
            for _ in range(steps):
                acceptance, delta_e, delta_m = metropolis_sweep(
                    state["spins"], temperature, state["replica_rng"], J=state["J"], h=state["h"]
                )
                state["energy_raw"] += delta_e
                state["magnetization_raw"] += delta_m
                self.time_energy.append(state["energy_raw"] / state["n_sites"])
                self.time_magnetization.append(state["magnetization_raw"] / state["n_sites"])
                self.time_acceptance.append(acceptance)
            state["phase_step"] += steps
            self.spins = state["spins"]
            self.progress_var.set(
                100.0
                * (
                    idx
                    + (
                        state["replica_index"]
                        + 0.5 * state["phase_step"] / max(1, state["local_equil"])
                    )
                    / state["replica_count"]
                )
                / total_temperatures
            )
            self.status_var.set(
                f"Equilibrating T = {temperature:.3f} ({idx + 1}/{total_temperatures}), "
                f"replica {state['replica_index'] + 1}/{state['replica_count']}, "
                f"sweep {state['phase_step']}/{state['local_equil']}"
            )
            self._update_summary()
            self._refresh_live_plots()
            if state["phase_step"] >= state["local_equil"]:
                state["phase"] = "measure"
                state["phase_step"] = 0
            self.root.after(1, self._run_step)
            return

        if state["phase"] == "measure":
            remaining = state["local_meas"] - state["phase_step"]
            steps = min(batch_size, remaining)
            for _ in range(steps):
                acceptance, delta_e, delta_m = metropolis_sweep(
                    state["spins"], temperature, state["replica_rng"], J=state["J"], h=state["h"]
                )
                state["energy_raw"] += delta_e
                state["magnetization_raw"] += delta_m
                energy = state["energy_raw"] / state["n_sites"]
                magnetization = state["magnetization_raw"] / state["n_sites"]
                state["energy_sum_total"] += state["energy_raw"]
                state["energy_sq_sum_total"] += state["energy_raw"] ** 2
                state["magnet_sum_total"] += state["magnetization_raw"]
                state["magnet_abs_sum_total"] += abs(state["magnetization_raw"])
                state["magnet_sq_sum_total"] += state["magnetization_raw"] ** 2
                state["measurement_count_total"] += 1
                if state["track_correlation"]:
                    state["correlation_sum_total"] += spin_correlation_profile(state["spins"])
                self.time_energy.append(energy)
                self.time_magnetization.append(magnetization)
                self.time_acceptance.append(acceptance)
            state["phase_step"] += steps
            self.spins = state["spins"]
            self.progress_var.set(
                100.0
                * (
                    idx
                    + (state["replica_index"] + 0.5 + 0.5 * state["phase_step"] / state["local_meas"])
                    / state["replica_count"]
                )
                / total_temperatures
            )
            self.status_var.set(
                f"Measuring T = {temperature:.3f} ({idx + 1}/{total_temperatures}), "
                f"replica {state['replica_index'] + 1}/{state['replica_count']}, "
                f"sweep {state['phase_step']}/{state['local_meas']}"
            )
            self._update_summary()
            self._refresh_live_plots()
            if state["phase_step"] >= state["local_meas"]:
                if state["replica_index"] + 1 < state["replica_count"]:
                    state["replica_index"] += 1
                    state["phase"] = "equil"
                    state["phase_step"] = 0
                    replica_rng = np.random.default_rng(self.rng.integers(0, 2**63 - 1))
                    state["replica_rng"] = replica_rng
                    state["spins"] = initialize_lattice(
                        state["L"], start=state["start_mode"], rng=replica_rng
                    )
                    self.spins = state["spins"]
                    self.current_temperature = temperature
                    state["energy_raw"] = total_energy(state["spins"], J=state["J"], h=state["h"])
                    state["magnetization_raw"] = float(total_magnetization(state["spins"]))
                    self.time_energy = [state["energy_raw"] / state["n_sites"]]
                    self.time_magnetization = [state["magnetization_raw"] / state["n_sites"]]
                    self.time_acceptance = [0.0]
                else:
                    n_sites = state["n_sites"]
                    mean_energy_raw = state["energy_sum_total"] / state["measurement_count_total"]
                    mean_magnet_raw = state["magnet_sum_total"] / state["measurement_count_total"]
                    mean_energy_sq = state["energy_sq_sum_total"] / state["measurement_count_total"]
                    mean_magnet_sq = state["magnet_sq_sum_total"] / state["measurement_count_total"]
                    mean_energy = mean_energy_raw / n_sites
                    mean_magnet = mean_magnet_raw / n_sites
                    mean_abs_magnet = state["magnet_abs_sum_total"] / (
                        state["measurement_count_total"] * n_sites
                    )

                    if temperature > 0.0:
                        cv = (mean_energy_sq - mean_energy_raw**2) / (n_sites * temperature**2)
                        chi = (mean_magnet_sq - mean_magnet_raw**2) / (n_sites * temperature)
                    else:
                        chi = 0.0
                        cv = 0.0

                    self.scan_rows.append(
                        {
                            "temperature": temperature,
                            "avg_energy_per_spin": float(mean_energy),
                            "avg_magnetization_per_spin": float(mean_magnet),
                            "avg_abs_magnetization_per_spin": float(mean_abs_magnet),
                            "heat_capacity": float(cv),
                            "susceptibility": float(chi),
                        }
                    )
                    if state["track_correlation"]:
                        connected_corr = (
                            state["correlation_sum_total"] / state["measurement_count_total"]
                            - mean_magnet**2
                        )
                        self.scan_correlations.append(
                            {
                                "temperature": temperature,
                                "profile": connected_corr,
                            }
                        )

                    state["temperature_index"] += 1
                    state["phase"] = "setup"
                    self.status_var.set(
                        f"Finished T = {temperature:.3f} ({state['temperature_index']}/{total_temperatures})"
                    )
                self._update_summary()
                self._refresh_plots()
            self.root.after(1, self._run_step)
            return

    def _finish_run(self, error_message):
        self.run_in_progress = False
        self.run_paused = False
        self.run_state = None
        self.current_temperature = None
        self._update_button_states()
        self._update_summary()
        self._refresh_plots()
        if error_message == "stopped":
            self.status_var.set("Run stopped.")
            return
        if error_message:
            self.status_var.set("Run failed.")
            messagebox.showerror("Run Failed", error_message)
            return
        self.progress_var.set(100.0 if self.scan_rows else 0.0)
        self.status_var.set("Run complete.")

    def export_scan_csv(self):
        if not self.scan_rows:
            messagebox.showinfo("No Data", "Press Run before exporting CSV data.")
            return

        path = filedialog.asksaveasfilename(
            title="Save scan data",
            defaultextension=".csv",
            initialfile=Path("thermodynamic_scan.csv").name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.scan_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.scan_rows)

        self.status_var.set(f"Saved scan data to {path}")
        self._update_summary()


def main():
    root = tk.Tk()
    IsingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

"""Lightweight Tkinter GUI for PMN inference.

This desktop app is intentionally minimal so it works on a laptop GPU:
- choose input/model/output paths
- toggle RAW auto-ratio and tiled inference
- adjust RGB strength live
- run inference in a background thread
- preview the saved denoised result and comparison image
"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import cv2

import run_inference as ri


APP_TITLE = 'PMN Low-Light Denoiser'
SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((SCRIPT_DIR / path).resolve())


def _image_to_tk(path: str, max_size: tuple[int, int] = (520, 360)) -> tk.PhotoImage:
    """Load an image as a Tk PhotoImage using OpenCV only."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    h, w = img.shape[:2]
    max_w, max_h = max_size
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    temp_path = str(Path(path).with_suffix('.gui_preview.png'))
    cv2.imwrite(temp_path, img)
    return tk.PhotoImage(file=temp_path)


class PMNGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry('1180x760')
        self.minsize(1040, 680)

        self.job_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._running = False
        self._input_preview: tk.PhotoImage | None = None
        self._output_preview: tk.PhotoImage | None = None
        self._comparison_preview: tk.PhotoImage | None = None

        self.input_var = tk.StringVar(value=str(SCRIPT_DIR / 'test_image' / 'RBB_noise.jpg'))
        self.model_var = tk.StringVar(value=str(SCRIPT_DIR / 'checkpoints' / 'SonyA7S2_Mix_Unet_best_model.pth'))
        self.output_var = tk.StringVar(value=str(SCRIPT_DIR / 'results_gui'))
        self.ratio_var = tk.DoubleVar(value=250.0)
        self.strength_var = tk.DoubleVar(value=0.4)
        self.tile_var = tk.IntVar(value=768)
        self.tile_overlap_var = tk.IntVar(value=64)
        self.auto_ratio_var = tk.BooleanVar(value=True)
        self.target_luma_var = tk.DoubleVar(value=0.18)
        self.save_maps_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value='Choose an image and run inference.')

        self._build_ui()
        self.after(150, self._poll_queue)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=14)
        controls.grid(row=0, column=0, sticky='nsw')
        controls.columnconfigure(1, weight=1)

        preview = ttk.Frame(self, padding=(0, 14, 14, 14))
        preview.grid(row=0, column=1, sticky='nsew')
        preview.columnconfigure(0, weight=1)
        preview.columnconfigure(1, weight=1)
        preview.rowconfigure(1, weight=1)

        title = ttk.Label(controls, text='PMN Low-Light Denoiser', font=('Segoe UI', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 10))

        self._path_row(controls, 1, 'Input', self.input_var, self._browse_input)
        self._path_row(controls, 2, 'Model', self.model_var, self._browse_model)
        self._path_row(controls, 3, 'Output', self.output_var, self._browse_output)

        sep = ttk.Separator(controls)
        sep.grid(row=4, column=0, columnspan=3, sticky='ew', pady=10)

        ttk.Checkbutton(controls, text='Auto ratio for RAW', variable=self.auto_ratio_var).grid(row=5, column=0, columnspan=2, sticky='w')
        ttk.Checkbutton(controls, text='Save diagnostic maps', variable=self.save_maps_var).grid(row=6, column=0, columnspan=2, sticky='w')

        self._slider_row(controls, 7, 'RAW ratio', self.ratio_var, 50, 400, 1)
        self._slider_row(controls, 8, 'RGB strength', self.strength_var, 0.0, 1.0, 0.01)
        self._slider_row(controls, 9, 'Tile size', self.tile_var, 0, 1536, 32)
        self._slider_row(controls, 10, 'Tile overlap', self.tile_overlap_var, 0, 256, 8)
        self._slider_row(controls, 11, 'Target luma', self.target_luma_var, 0.05, 0.35, 0.01)

        button_row = ttk.Frame(controls)
        button_row.grid(row=12, column=0, columnspan=3, sticky='ew', pady=(10, 4))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)

        self.run_button = ttk.Button(button_row, text='Run Inference', command=self._start_inference)
        self.run_button.grid(row=0, column=0, sticky='ew', padx=(0, 4))

        ttk.Button(button_row, text='Load Preview', command=self._load_previews).grid(row=0, column=1, sticky='ew', padx=(4, 0))

        self.progress = ttk.Progressbar(controls, mode='indeterminate')
        self.progress.grid(row=13, column=0, columnspan=3, sticky='ew', pady=(10, 4))

        status = ttk.Label(controls, textvariable=self.status_var, wraplength=360, foreground='#264653')
        status.grid(row=14, column=0, columnspan=3, sticky='ew', pady=(8, 0))

        self.original_label = ttk.LabelFrame(preview, text='Original')
        self.original_label.grid(row=0, column=0, sticky='nsew', padx=(0, 8), pady=(0, 8))
        self.original_label.rowconfigure(0, weight=1)
        self.original_label.columnconfigure(0, weight=1)

        self.output_label = ttk.LabelFrame(preview, text='Denoised Output')
        self.output_label.grid(row=0, column=1, sticky='nsew', padx=(8, 0), pady=(0, 8))
        self.output_label.rowconfigure(0, weight=1)
        self.output_label.columnconfigure(0, weight=1)

        self.comparison_label = ttk.LabelFrame(preview, text='Comparison')
        self.comparison_label.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=(8, 0))
        self.comparison_label.rowconfigure(0, weight=1)
        self.comparison_label.columnconfigure(0, weight=1)

        self.original_canvas = ttk.Label(self.original_label, anchor='center')
        self.original_canvas.grid(row=0, column=0, sticky='nsew', padx=8, pady=8)
        self.output_canvas = ttk.Label(self.output_label, anchor='center')
        self.output_canvas.grid(row=0, column=0, sticky='nsew', padx=8, pady=8)
        self.comparison_canvas = ttk.Label(self.comparison_label, anchor='center')
        self.comparison_canvas.grid(row=0, column=0, sticky='nsew', padx=8, pady=8)

    def _path_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, callback) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=3)
        entry = ttk.Entry(parent, textvariable=variable, width=44)
        entry.grid(row=row, column=1, sticky='ew', padx=(8, 8), pady=3)
        ttk.Button(parent, text='Browse', command=callback).grid(row=row, column=2, sticky='ew', pady=3)

    def _slider_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable, from_: float, to: float, resolution: float) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=(8, 0))
        slider = ttk.Scale(parent, from_=from_, to=to, variable=variable)
        slider.grid(row=row, column=1, sticky='ew', padx=(8, 8), pady=(8, 0))

        value_label = ttk.Label(parent, width=8)
        value_label.grid(row=row, column=2, sticky='e', pady=(8, 0))

        def refresh_value(*_):
            value = variable.get()
            if isinstance(variable, tk.IntVar):
                value_label.config(text=str(int(value)))
            else:
                if resolution >= 0.1:
                    value_label.config(text=f'{float(value):.2f}')
                else:
                    value_label.config(text=f'{float(value):.3f}')

        refresh_value()
        variable.trace_add('write', refresh_value)

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(title='Select input image')
        if path:
            self.input_var.set(path)
            self._load_previews()

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(title='Select model checkpoint', filetypes=[('PyTorch checkpoint', '*.pth *.pt'), ('All files', '*.*')])
        if path:
            self.model_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title='Select output folder')
        if path:
            self.output_var.set(path)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _load_previews(self) -> None:
        try:
            input_path = _resolve_path(self.input_var.get())
            if not os.path.exists(input_path):
                raise FileNotFoundError(input_path)

            self._input_preview = _image_to_tk(input_path)
            self.original_canvas.config(image=self._input_preview)
            self._set_status(f'Loaded preview for {os.path.basename(input_path)}')
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f'Preview failed: {exc}')

    def _start_inference(self) -> None:
        if self._running:
            return

        self._running = True
        self.run_button.config(state='disabled')
        self.progress.start(10)
        self._set_status('Running inference...')

        thread = threading.Thread(target=self._run_worker, daemon=True)
        thread.start()

    def _run_worker(self) -> None:
        try:
            input_path = _resolve_path(self.input_var.get())
            model_path = _resolve_path(self.model_var.get())
            output_dir = _resolve_path(self.output_var.get())

            result = ri.run(
                image_path=input_path,
                model_path=model_path,
                output_dir=output_dir,
                ratio=float(self.ratio_var.get()),
                strength=float(self.strength_var.get()),
                auto_ratio=bool(self.auto_ratio_var.get()),
                target_luma=float(self.target_luma_var.get()),
                tile=int(self.tile_var.get()),
                tile_overlap=int(self.tile_overlap_var.get()),
                save_maps=bool(self.save_maps_var.get()),
                return_results=True,
            )

            stem = result['stem']
            output_image = os.path.join(output_dir, f'{stem}_denoised.png')
            comparison_image = os.path.join(output_dir, f'{stem}_comparison.png')

            self.job_queue.put(('done', (output_image, comparison_image, output_dir)))
        except Exception as exc:
            self.job_queue.put(('error', str(exc)))

    def _poll_queue(self) -> None:
        try:
            item = self.job_queue.get_nowait()
        except queue.Empty:
            self.after(150, self._poll_queue)
            return

        kind, payload = item
        if kind == 'error':
            self._running = False
            self.progress.stop()
            self.run_button.config(state='normal')
            self._set_status('Inference failed.')
            messagebox.showerror(APP_TITLE, str(payload))
            self.after(150, self._poll_queue)
            return

        output_image, comparison_image, output_dir = payload
        try:
            self._output_preview = _image_to_tk(output_image)
            self.output_canvas.config(image=self._output_preview)

            if os.path.exists(comparison_image):
                self._comparison_preview = _image_to_tk(comparison_image, (1100, 340))
                self.comparison_canvas.config(image=self._comparison_preview)

            self._set_status(f'Finished. Results saved to {output_dir}')
        except Exception as exc:
            messagebox.showwarning(APP_TITLE, f'Inference finished, but preview loading failed: {exc}')
            self._set_status(f'Finished. Results saved to {output_dir}')
        finally:
            self._running = False
            self.progress.stop()
            self.run_button.config(state='normal')

        self.after(150, self._poll_queue)


def main() -> None:
    app = PMNGUI()
    app.mainloop()


if __name__ == '__main__':
    main()
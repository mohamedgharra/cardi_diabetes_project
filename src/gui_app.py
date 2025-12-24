"""Tkinter-based GUI to explore data, train models, and run inference interactively."""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox, ttk

from . import config, infer, run_all

TEXT_FILE_EXTENSIONS = {".py", ".md", ".txt", ".csv", ".json", ".yml", ".yaml"}


class CardioDiabetesGUI(ttk.Frame):
    """Interactive desktop companion similar to Orange's Test & Score panel."""

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master
        self.pack(fill="both", expand=True)
        self.current_file: Optional[Path] = None
        self.dataset_cache: Dict[str, pd.DataFrame] = {}
        self.custom_dataset_path: Optional[Path] = None
        self.predictions_df: Optional[pd.DataFrame] = None
        self.training_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None

        self._build_ui()
        self._populate_file_tree()
        self.after(200, self._refresh_metrics_table)

    # ------------------------------------------------------------------
    # UI layout helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.files_tab = ttk.Frame(notebook)
        self.data_tab = ttk.Frame(notebook)
        self.training_tab = ttk.Frame(notebook)
        self.infer_tab = ttk.Frame(notebook)

        notebook.add(self.files_tab, text="File & Editor")
        notebook.add(self.data_tab, text="Dati & Grafici")
        notebook.add(self.training_tab, text="Training & Score")
        notebook.add(self.infer_tab, text="Inferenza")

        self._build_file_tab()
        self._build_data_tab()
        self._build_training_tab()
        self._build_infer_tab()

    # ------------------------------------------------------------------
    # File manager tab
    # ------------------------------------------------------------------
    def _build_file_tab(self) -> None:
        container = ttk.Panedwindow(self.files_tab, orient=tk.HORIZONTAL)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(container)
        right_frame = ttk.Frame(container)
        container.add(left_frame, weight=1)
        container.add(right_frame, weight=3)

        self.file_tree = ttk.Treeview(left_frame, columns=("path",), show="tree")
        self.file_tree.pack(fill="both", expand=True)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Apri", command=self._open_selected_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Aggiorna", command=self._populate_file_tree).pack(side=tk.LEFT, padx=2)

        editor_frame = ttk.Frame(right_frame)
        editor_frame.pack(fill="both", expand=True)
        self.file_text = tk.Text(editor_frame, wrap="none")
        self.file_text.pack(fill="both", expand=True)

        save_frame = ttk.Frame(right_frame)
        save_frame.pack(fill="x", pady=5)
        ttk.Button(save_frame, text="Salva", command=self._save_current_file).pack(side=tk.RIGHT)

    def _populate_file_tree(self) -> None:
        for child in self.file_tree.get_children():
            self.file_tree.delete(child)

        root = config.PROJECT_ROOT
        root_node = self.file_tree.insert("", tk.END, text=root.name, open=True, values=(str(root),))
        self._add_tree_children(root_node, root, depth=0)

    def _add_tree_children(self, parent, path: Path, depth: int) -> None:
        if depth > 3:
            return
        for child in sorted(path.iterdir()):
            if child.name.startswith(".") or child.name == "__pycache__":
                continue
            node = self.file_tree.insert(parent, tk.END, text=child.name, values=(str(child),))
            if child.is_dir():
                self._add_tree_children(node, child, depth + 1)

    def _open_selected_file(self) -> None:
        selection = self.file_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Seleziona un file dall'elenco")
            return
        path = Path(self.file_tree.item(selection[0], "values")[0])
        if path.is_dir():
            messagebox.showinfo("Info", "Seleziona un file, non una cartella")
            return
        if path.suffix.lower() not in TEXT_FILE_EXTENSIONS:
            messagebox.showwarning("Avviso", "Solo file di testo modificabili direttamente qui")
            return
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            messagebox.showerror("Errore", "Impossibile leggere il file come testo UTF-8")
            return
        self.current_file = path
        self.file_text.delete("1.0", tk.END)
        self.file_text.insert(tk.END, content)

    def _save_current_file(self) -> None:
        if not self.current_file:
            messagebox.showinfo("Info", "Nessun file aperto da salvare")
            return
        try:
            self.current_file.write_text(self.file_text.get("1.0", tk.END), encoding="utf-8")
            messagebox.showinfo("Successo", f"File salvato: {self.current_file}")
        except OSError as exc:
            messagebox.showerror("Errore", f"Impossibile salvare il file: {exc}")

    # ------------------------------------------------------------------
    # Data exploration tab
    # ------------------------------------------------------------------
    def _build_data_tab(self) -> None:
        top_frame = ttk.Frame(self.data_tab)
        top_frame.pack(fill="x", padx=10, pady=10)

        self.dataset_var = tk.StringVar(value="train")
        ttk.Label(top_frame, text="Dataset").grid(row=0, column=0, sticky="w")
        dataset_combo = ttk.Combobox(
            top_frame,
            textvariable=self.dataset_var,
            values=["train", "test", "custom"],
            state="readonly",
            width=10,
        )
        dataset_combo.grid(row=0, column=1, padx=5)
        dataset_combo.bind("<<ComboboxSelected>>", lambda _: self._refresh_column_list())

        ttk.Button(top_frame, text="Carica file personalizzato", command=self._choose_custom_dataset).grid(
            row=0, column=2, padx=5
        )
        ttk.Button(top_frame, text="Mostra anteprima", command=self._show_dataset_preview).grid(
            row=0, column=3, padx=5
        )

        ttk.Label(top_frame, text="Grafico").grid(row=1, column=0, sticky="w", pady=5)
        self.graph_type_var = tk.StringVar(value="Distribuzione")
        graph_combo = ttk.Combobox(
            top_frame,
            textvariable=self.graph_type_var,
            values=["Distribuzione", "Violin target", "Correlazione"],
            state="readonly",
            width=18,
        )
        graph_combo.grid(row=1, column=1, padx=5, sticky="w")

        ttk.Label(top_frame, text="Colonne").grid(row=2, column=0, sticky="nw")
        self.column_listbox = tk.Listbox(top_frame, selectmode=tk.MULTIPLE, height=6)
        self.column_listbox.grid(row=2, column=1, columnspan=2, sticky="nsew", pady=5)
        top_frame.grid_columnconfigure(1, weight=1)

        ttk.Button(top_frame, text="Genera grafico", command=self._generate_graph).grid(row=2, column=3, padx=5)

        # Preview table
        preview_frame = ttk.Frame(self.data_tab)
        preview_frame.pack(fill="x", padx=10)
        self.preview_text = tk.Text(preview_frame, height=10)
        self.preview_text.pack(fill="x", expand=True)

        # Graph area
        graph_frame = ttk.Frame(self.data_tab)
        graph_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.graph_canvas_container = graph_frame
        self.graph_canvas: Optional[FigureCanvasTkAgg] = None

        self._refresh_column_list()

    def _get_dataset_path(self) -> Optional[Path]:
        choice = self.dataset_var.get()
        if choice == "train":
            return config.DATA_DIR / "train.csv"
        if choice == "test":
            return config.DATA_DIR / "test.csv"
        return self.custom_dataset_path

    def _load_dataset(self, path: Path) -> pd.DataFrame:
        cache_key = str(path)
        if cache_key in self.dataset_cache:
            return self.dataset_cache[cache_key]
        df = pd.read_csv(path)
        self.dataset_cache[cache_key] = df
        return df

    def _choose_custom_dataset(self) -> None:
        selected = filedialog.askopenfilename(title="Seleziona un CSV", filetypes=[("CSV", "*.csv")])
        if selected:
            self.custom_dataset_path = Path(selected)
            self.dataset_var.set("custom")
            self._refresh_column_list()

    def _refresh_column_list(self) -> None:
        path = self._get_dataset_path()
        self.column_listbox.delete(0, tk.END)
        if not path or not path.exists():
            return
        df = self._load_dataset(path)
        for col in df.columns:
            self.column_listbox.insert(tk.END, col)

    def _show_dataset_preview(self) -> None:
        path = self._get_dataset_path()
        if not path or not path.exists():
            messagebox.showerror("Errore", "File dataset non trovato")
            return
        df = self._load_dataset(path)
        preview = df.head(15).to_string()
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert(tk.END, preview)

    def _generate_graph(self) -> None:
        path = self._get_dataset_path()
        if not path or not path.exists():
            messagebox.showerror("Errore", "File dataset non trovato")
            return
        df = self._load_dataset(path)
        selection = [self.column_listbox.get(i) for i in self.column_listbox.curselection()]
        graph_type = self.graph_type_var.get()

        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        try:
            if graph_type == "Correlazione":
                corr = df.corr(numeric_only=True)
                ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(len(corr.columns)))
                ax.set_yticklabels(corr.columns, fontsize=8)
                ax.set_title("Heatmap correlazione")
            elif graph_type == "Violin target":
                target = config.TARGET if config.TARGET in df.columns else None
                if not target:
                    raise ValueError("Il dataset selezionato non contiene la colonna target")
                if not selection:
                    raise ValueError("Seleziona almeno una colonna per il grafico")
                col = selection[0]
                categories = df[target].astype(str)
                ax.set_title(f"Violin plot: {col} vs {target}")
                ax.violinplot(
                    [df.loc[categories == level, col].dropna() for level in categories.unique()],
                    showmeans=True,
                )
                ax.set_xticks(range(1, len(categories.unique()) + 1))
                ax.set_xticklabels(categories.unique())
                ax.set_ylabel(col)
            else:  # Distribuzione
                if not selection:
                    raise ValueError("Seleziona almeno una colonna per il grafico")
                for col in selection:
                    ax.hist(df[col].dropna(), bins=30, alpha=0.6, label=col)
                ax.set_title("Distribuzioni selezionate")
                ax.legend()
        except Exception as exc:
            messagebox.showerror("Errore", f"Impossibile generare il grafico: {exc}")
            return

        fig.tight_layout()
        if self.graph_canvas:
            self.graph_canvas.get_tk_widget().destroy()
        self.graph_canvas = FigureCanvasTkAgg(fig, master=self.graph_canvas_container)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Training tab
    # ------------------------------------------------------------------
    def _build_training_tab(self) -> None:
        options_frame = ttk.LabelFrame(self.training_tab, text="Opzioni training")
        options_frame.pack(fill="x", padx=10, pady=10)

        self.holdout_var = tk.BooleanVar(value=True)
        self.cv5_var = tk.BooleanVar(value=True)
        self.cv5_tuning_var = tk.BooleanVar(value=True)
        self.weight_var = tk.BooleanVar(value=False)
        self.seed_var = tk.StringVar(value=str(config.SEED))

        ttk.Checkbutton(options_frame, text="Hold-out", variable=self.holdout_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="CV 5-fold", variable=self.cv5_var).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(options_frame, text="CV 5-fold + tuning", variable=self.cv5_tuning_var).grid(
            row=0, column=2, sticky="w"
        )
        ttk.Checkbutton(options_frame, text="Peso feature prioritare", variable=self.weight_var).grid(
            row=1, column=0, sticky="w", pady=5
        )
        ttk.Label(options_frame, text="Seed").grid(row=1, column=1, sticky="e")
        ttk.Entry(options_frame, textvariable=self.seed_var, width=8).grid(row=1, column=2, sticky="w")

        ttk.Button(options_frame, text="Avvia training", command=self._trigger_training).grid(
            row=0,
            column=3,
            rowspan=2,
            padx=10,
            sticky="ns",
        )

        log_frame = ttk.LabelFrame(self.training_tab, text="Log e stato")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.training_log = tk.Text(log_frame, height=10)
        self.training_log.pack(fill="both", expand=True)

        table_frame = ttk.LabelFrame(self.training_tab, text="Test & Score (riassunto)")
        table_frame.pack(fill="both", expand=True, padx=10, pady=5)
        columns = ("model", "technique", "roc_auc", "f1", "accuracy")
        self.metrics_table = ttk.Treeview(table_frame, columns=columns, show="headings")
        for col in columns:
            self.metrics_table.heading(col, text=col)
        self.metrics_table.pack(fill="both", expand=True)

    def _trigger_training(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("In corso", "Il training è già in esecuzione")
            return

        techniques: List[str] = []
        if self.holdout_var.get():
            techniques.append("holdout")
        if self.cv5_var.get():
            techniques.append("cv5")
        if self.cv5_tuning_var.get():
            techniques.append("cv5_tuning")
        if not techniques:
            messagebox.showerror("Errore", "Seleziona almeno una tecnica di validazione")
            return

        try:
            seed = int(self.seed_var.get())
        except ValueError:
            messagebox.showerror("Errore", "Seed non valido")
            return

        self.training_log.delete("1.0", tk.END)
        self.training_log.insert(tk.END, "Training avviato...\n")

        def worker():
            try:
                result = run_all.run_pipeline(
                    data_train=str(config.DATA_DIR / "train.csv"),
                    data_test=str(config.DATA_DIR / "test.csv"),
                    techniques=techniques,
                    weight_top_features=self.weight_var.get(),
                    seed=seed,
                    progress_callback=lambda msg: self._append_log(msg),
                )
                self.after(0, lambda: self._on_training_finished(result))
            except Exception as exc:  # pragma: no cover - GUI feedback
                self.after(0, lambda: messagebox.showerror("Errore", str(exc)))
                self._append_log(f"Errore: {exc}")

        self.training_thread = threading.Thread(target=worker, daemon=True)
        self.training_thread.start()

    def _append_log(self, message: str) -> None:
        def _update():
            self.training_log.insert(tk.END, message + "\n")
            self.training_log.see(tk.END)

        self.training_log.after(0, _update)

    def _on_training_finished(self, result: Dict[str, object]) -> None:
        self._append_log("Training completato")
        self._load_metrics_table(result.get("metrics_path"))
        best_info = (
            f"Miglior modello: {result.get('best_model')} ({result.get('best_technique')}) "
            f"ROC-AUC={result.get('best_score'):.3f}"
        )
        self._append_log(best_info)
        messagebox.showinfo("Completato", best_info)

    def _refresh_metrics_table(self) -> None:
        metrics_path = config.TABLES_DIR / "summary_metrics.csv"
        if metrics_path.exists():
            self._load_metrics_table(metrics_path)
        self.after(5000, self._refresh_metrics_table)

    def _load_metrics_table(self, metrics_path: Optional[Path | str]) -> None:
        if metrics_path is None:
            return
        path = Path(metrics_path)
        if not path.exists():
            return
        df = pd.read_csv(path)
        if df.empty:
            return
        df = df.sort_values("roc_auc_mean", ascending=False)
        for row in self.metrics_table.get_children():
            self.metrics_table.delete(row)
        for _, row in df.iterrows():
            self.metrics_table.insert(
                "",
                tk.END,
                values=(
                    row["model"],
                    row["technique"],
                    f"{row.get('roc_auc_mean', 0):.3f}",
                    f"{row.get('f1_macro_mean', 0):.3f}",
                    f"{row.get('accuracy_mean', 0):.3f}",
                ),
            )

    # ------------------------------------------------------------------
    # Inference tab
    # ------------------------------------------------------------------
    def _build_infer_tab(self) -> None:
        form = ttk.Frame(self.infer_tab)
        form.pack(fill="x", padx=10, pady=10)

        ttk.Label(form, text="Modello").grid(row=0, column=0, sticky="e")
        self.model_path_var = tk.StringVar(value=str(config.MODELS_DIR / "best_model.joblib"))
        ttk.Entry(form, textvariable=self.model_path_var, width=60).grid(row=0, column=1, padx=5, sticky="ew")
        ttk.Button(form, text="Sfoglia", command=self._choose_model).grid(row=0, column=2)

        ttk.Label(form, text="Dataset di test").grid(row=1, column=0, sticky="e")
        self.test_path_var = tk.StringVar(value=str(config.DATA_DIR / "test.csv"))
        ttk.Entry(form, textvariable=self.test_path_var, width=60).grid(row=1, column=1, padx=5, sticky="ew")
        ttk.Button(form, text="Sfoglia", command=self._choose_test_file).grid(row=1, column=2)

        ttk.Button(form, text="Genera predizioni", command=self._trigger_inference).grid(
            row=2, column=0, columnspan=3, pady=5
        )

        table_frame = ttk.LabelFrame(self.infer_tab, text="Anteprima predizioni")
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)
        columns = ("id", "pred", "prob")
        self.pred_table = ttk.Treeview(table_frame, columns=columns, show="headings")
        for col in columns:
            self.pred_table.heading(col, text=col)
        self.pred_table.pack(fill="both", expand=True)

        ttk.Button(self.infer_tab, text="Esporta CSV", command=self._export_predictions).pack(pady=5)

    def _choose_model(self) -> None:
        selected = filedialog.askopenfilename(title="Seleziona modello", filetypes=[("joblib", "*.joblib")])
        if selected:
            self.model_path_var.set(selected)

    def _choose_test_file(self) -> None:
        selected = filedialog.askopenfilename(title="Seleziona CSV", filetypes=[("CSV", "*.csv")])
        if selected:
            self.test_path_var.set(selected)

    def _trigger_inference(self) -> None:
        if self.inference_thread and self.inference_thread.is_alive():
            messagebox.showinfo("In corso", "Inferenza già in esecuzione")
            return

        model_path = self.model_path_var.get()
        data_path = self.test_path_var.get()
        if not Path(model_path).exists():
            messagebox.showerror("Errore", "File modello non trovato")
            return
        if not Path(data_path).exists():
            messagebox.showerror("Errore", "File di test non trovato")
            return

        def worker():
            try:
                preds = infer.run_inference(model_path, data_path)
                self.predictions_df = preds
                self.after(0, lambda: self._display_predictions(preds))
                self.after(0, lambda: messagebox.showinfo("Inferenza", f"Predizioni generate: {len(preds)}"))
            except Exception as exc:  # pragma: no cover - UI feedback
                self.after(0, lambda: messagebox.showerror("Errore", str(exc)))

        self.inference_thread = threading.Thread(target=worker, daemon=True)
        self.inference_thread.start()

    def _display_predictions(self, preds: pd.DataFrame) -> None:
        for row in self.pred_table.get_children():
            self.pred_table.delete(row)
        preview = preds.head(50)
        for _, row in preview.iterrows():
            self.pred_table.insert(
                "",
                tk.END,
                values=(row["id"], int(row["diabetes_pred"]), f"{row['prob_diabetes']:.3f}"),
            )

    def _export_predictions(self) -> None:
        if self.predictions_df is None or self.predictions_df.empty:
            messagebox.showinfo("Info", "Nessuna predizione da esportare")
            return
        target = filedialog.asksaveasfilename(
            title="Salva CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if target:
            self.predictions_df.to_csv(target, index=False)
            messagebox.showinfo("Successo", f"Predizioni salvate in {target}")


def launch_app() -> None:
    root = tk.Tk()
    root.title("Cardio Diabetes Studio")
    root.geometry("1200x800")
    CardioDiabetesGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_app()

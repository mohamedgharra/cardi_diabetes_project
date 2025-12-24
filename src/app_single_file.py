"""Single-file Streamlit GUI for the cardio_diabetes_project.

This app lets you:
- load the project's example data or upload your own CSV
- preview and inspect data (table, summary, columns)
- choose simple plots (histogram, scatter, boxplot, correlation heatmap)
- load a trained model (default: `reports/models/best_model.joblib`) or upload one
- run inference on the loaded table, view predictions, download results
- when the dataset contains the target column (`diabetes`) the app computes classification metrics

Run with (from project inner folder):
    python -m streamlit run src/app_single_file.py

This file intentionally tries to import the project's modules by adding the project
package folder to sys.path so it works when placed inside `src/`.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _ensure_project_importable():
    # When running this file directly, make sure the parent folder of `src` is on sys.path
    here = Path(__file__).resolve()
    project_inner = here.parents[1]  # cardio_diabetes_project (inner)
    if str(project_inner) not in sys.path:
        sys.path.insert(0, str(project_inner))


_ensure_project_importable()

try:
    # import project modules
    from src import config, preprocessing, evaluation, models, train_validate, utils_io
except Exception:
    # Best-effort: if imports fail, set these to None and continue with reduced functionality
    config = None
    preprocessing = None
    evaluation = None
    models = None
    train_validate = None
    utils_io = None


st.set_page_config(page_title="Cardio Diabetes - Mini GUI", layout="wide")

# Small visual polish: custom CSS to increase font and make header nicer
st.markdown(
    """
    <style>
    .stApp { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .big-title { font-size:28px; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_csv_from_path(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_model_from_path(path: Path):
    return joblib.load(path)


def infer_predictions(estimator, df: pd.DataFrame):
    # If model is a pipeline it will do preprocessing internally; drop id if present only for display
    X = df.copy()
    y_true = None
    if config and config.TARGET in X.columns:
        y_true = X[config.TARGET].to_numpy()
    # Let evaluation.extract_probabilities handle predict/prob
    if evaluation is not None:
        try:
            y_pred, y_score = evaluation.extract_probabilities(estimator, X)
        except Exception:
            # fallback to plain predict if something goes wrong
            y_pred = estimator.predict(X)
            y_score = None
    else:
        y_pred = estimator.predict(X)
        y_score = None
    return y_true, y_pred, y_score


def initialize_session_state():
    """Initialize session state variables for persisting data across reruns."""
    if "test_score_results" not in st.session_state:
        st.session_state.test_score_results = None
    if "test_score_models" not in st.session_state:
        st.session_state.test_score_models = None
    if "test_score_techniques" not in st.session_state:
        st.session_state.test_score_techniques = None
    if "test_score_seed" not in st.session_state:
        st.session_state.test_score_seed = None
    if "test_score_details" not in st.session_state:
        st.session_state.test_score_details = {}
    if "final_prediction_table" not in st.session_state:
        st.session_state.final_prediction_table = None


def main():
    st.markdown('<div class="big-title">Cardio Diabetes — Mini GUI</div>', unsafe_allow_html=True)
    st.write("Una UI moderna per esplorare i dati, visualizzare grafici interattivi e eseguire inferenza con modelli salvati.")
    
    # Inizializza session state per persistere dati tra i rerun
    initialize_session_state()

    sidebar = st.sidebar
    sidebar.header("Input & Modello")
    use_example = sidebar.checkbox("Usa dataset di esempio (train.csv)", value=True)
    uploaded_file = sidebar.file_uploader("Oppure carica un file CSV", type=["csv"] )

    model_option = sidebar.radio("Sorgente modello", ("Default model file", "Carica modello (.joblib)"))
    default_model_path = Path(__file__).resolve().parents[1] / "reports" / "models" / "best_model.joblib"
    uploaded_model = None
    if model_option == "Carica modello (.joblib)":
        uploaded_model = sidebar.file_uploader("Carica modello joblib", type=["joblib", "pkl"]) 
    else:
        sidebar.write(f"Default: {default_model_path.name}")

    st.markdown("---")
    tabs = st.tabs(["Dati", "Grafici", "Inferenza", "Test & Score"])

    # Load dataframe (common)
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        file_note = f"Caricato {uploaded_file.name} ({len(df)} righe)"
    elif use_example and config is not None:
        example_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"
        try:
            df = load_csv_from_path(example_path)
            file_note = f"Usando dataset di esempio: {example_path.name} ({len(df)} righe)"
        except Exception as exc:
            st.error(f"Impossibile caricare il dataset di esempio: {exc}")
            df = pd.DataFrame()
            file_note = ""
    else:
        df = pd.DataFrame()
        file_note = ""

    # Tab: Dati
    with tabs[0]:
        st.subheader("Dati & Statistiche")
        if file_note:
            st.success(file_note)
        # Show dataset counts for train/test (project data folder)
        if config is not None and utils_io is not None:
            try:
                train_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"
                test_path = Path(__file__).resolve().parents[1] / "data" / "test.csv"
                train_rows = None
                test_rows = None
                try:
                    train_df_local = utils_io.load_csv(train_path)
                    train_rows = len(train_df_local)
                except Exception:
                    train_rows = None
                try:
                    test_df_local = utils_io.load_csv(test_path)
                    test_rows = len(test_df_local)
                except Exception:
                    test_rows = None
                st.markdown("**Dataset locali**")
                st.write({"train_rows": train_rows, "test_rows": test_rows})
                # Show last training summary if available
                try:
                    metrics_path = Path(__file__).resolve().parents[1] / "reports" / "tables" / "summary_metrics.csv"
                    if metrics_path.exists():
                        metrics_df = pd.read_csv(metrics_path)
                        if not metrics_df.empty:
                            best_row = metrics_df.sort_values("roc_auc_mean", ascending=False).iloc[0]
                            st.markdown("**Ultimo training (top result)**")
                            st.write({
                                "model": best_row.get("model"),
                                "technique": best_row.get("technique"),
                                "roc_auc_mean": best_row.get("roc_auc_mean"),
                                "f1_macro_mean": best_row.get("f1_macro_mean"),
                            })
                except Exception:
                    pass
            except Exception:
                pass

        if df.empty:
            st.info("Nessun dataset caricato. Carica un CSV o abilita il dataset di esempio nello sidebar.")
        else:
            # Mostra i dati locali disponibili (train e test di default)
            st.markdown("### Dataset locali disponibili")
            col_train, col_test = st.columns(2)
            with col_train:
                try:
                    train_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"
                    if train_path.exists():
                        train_local = pd.read_csv(train_path)
                        st.write(f"**Train set**: {len(train_local)} righe × {len(train_local.columns)} colonne")
                        with st.expander("Mostra train set completo"):
                            st.dataframe(train_local, use_container_width=True)
                except Exception as e:
                    st.warning(f"Impossibile caricare train.csv: {e}")
            
            with col_test:
                try:
                    test_path = Path(__file__).resolve().parents[1] / "data" / "test.csv"
                    if test_path.exists():
                        test_local = pd.read_csv(test_path)
                        st.write(f"**Test set**: {len(test_local)} righe × {len(test_local.columns)} colonne")
                        with st.expander("Mostra test set completo"):
                            st.dataframe(test_local, use_container_width=True)
                except Exception as e:
                    st.warning(f"Impossibile caricare test.csv: {e}")
            
            st.markdown("### Dataset caricato/selezionato")
            left, right = st.columns([2, 1])
            with left:
                # Mostra tutti i dati caricati, con opzione di limitare la visualizzazione
                n_rows = st.slider("Numero di righe da visualizzare", min_value=10, max_value=len(df), value=min(100, len(df)))
                st.dataframe(df.head(n_rows), use_container_width=True)
            with right:
                st.markdown("**Statistiche descrittive**")
                st.write(df.describe(include="all"))

    # Tab: Grafici
    with tabs[1]:
        st.subheader("Grafici interattivi")
        if df.empty:
            st.info("Carica un dataset per visualizzare i grafici interattivi")
        else:
            # feature selection
            if preprocessing is not None:
                try:
                    ordered = preprocessing.get_ordered_feature_list(df)
                except Exception:
                    ordered = list(df.columns)
            else:
                ordered = list(df.columns)

            selected = st.multiselect("Colonne", options=ordered, default=ordered[:6] if ordered else list(df.columns))
            plot_type = st.selectbox("Tipo grafico", ["Histogram", "Scatter (Plotly)", "Boxplot", "Correlation heatmap"]) 
            if plot_type == "Histogram":
                col = st.selectbox("Colonna per histogramma", options=selected)
                fig = px.histogram(df, x=col, color=(config.TARGET if config and config.TARGET in df.columns else None), marginal="box", nbins=40, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            elif plot_type == "Scatter (Plotly)":
                xcol = st.selectbox("X", options=selected, index=0)
                ycol = st.selectbox("Y", options=selected, index=min(1, len(selected) - 1))
                color_col = config.TARGET if config and config.TARGET in df.columns else None
                fig = px.scatter(df, x=xcol, y=ycol, color=color_col, template="plotly_white", hover_data=selected)
                st.plotly_chart(fig, use_container_width=True)
            elif plot_type == "Boxplot":
                col = st.selectbox("Colonna per boxplot", options=selected)
                fig = px.box(df, y=col, color=(config.TARGET if config and config.TARGET in df.columns else None), template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            elif plot_type == "Correlation heatmap":
                num = df.select_dtypes(include=["number"])
                corr = num.corr()
                fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu"))
                fig.update_layout(title="Correlation heatmap", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

    # Tab: Inferenza
    with tabs[2]:
        st.subheader("Inferenza e Test & Score")
        # Model loading
        model = None
        if model_option == "Carica modello (.joblib)" and uploaded_model is not None:
            try:
                model_bytes = uploaded_model.read()
                model = joblib.load(io.BytesIO(model_bytes))
                st.success("Modello caricato dall'upload")
            except Exception as exc:
                st.error(f"Errore nel caricare il modello: {exc}")
        else:
            if default_model_path.exists():
                try:
                    model = load_model_from_path(default_model_path)
                    st.success(f"Modello predefinito caricato: {default_model_path.name}")
                except Exception as exc:
                    st.error(f"Impossibile caricare il modello predefinito: {exc}")
            else:
                st.warning("Nessun modello predefinito trovato; carica un file .joblib nella sidebar")

        if df.empty:
            st.info("Carica un dataset per eseguire inferenza")
        elif model is None:
            st.info("Carica o specifica un modello per eseguire l'inferenza")
        else:
            if st.button("Esegui predizione sulle righe caricate"):
                try:
                    y_true, y_pred, y_score = infer_predictions(model, df)
                    out_df = df.copy()
                    out_df["diabetes_pred"] = y_pred
                    if y_score is not None:
                        out_df["prob_diabetes"] = y_score
                    
                    # Aggiungere label reale se disponibile
                    if y_true is not None:
                        out_df["diabetes_true"] = y_true
                        # Aggiungere colonna di errore (1 = errore, 0 = corretto)
                        out_df["is_error"] = (y_pred != y_true).astype(int)
                    
                    st.success("Predizioni completate")
                    
                    # Slider per selezionare numero di righe da visualizzare
                    n_show = st.slider("Righe da visualizzare", min_value=10, max_value=len(out_df), value=min(100, len(out_df)))
                    st.dataframe(out_df.head(n_show), use_container_width=True)
                    
                    # Statistiche su errori
                    if y_true is not None:
                        n_errors = (y_pred != y_true).sum()
                        accuracy = (y_pred == y_true).sum() / len(y_true)
                        st.info(f"Errori di predizione: {n_errors}/{len(y_true)} ({100*n_errors/len(y_true):.1f}%) | Accuracy: {accuracy:.3f}")


                    # Metrics & interactive plots
                    if y_true is not None and evaluation is not None:
                        metrics = evaluation.compute_classification_metrics(y_true, y_pred, y_score)
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                        col2.metric("F1 macro", f"{metrics.get('f1_macro', 0):.3f}")
                        col3.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}" if metrics.get('roc_auc') is not None else "n/a")

                        # Confusion matrix (plotly)
                        cm = pd.crosstab(y_true, y_pred)
                        cm_fig = px.imshow(cm.values, x=cm.columns, y=cm.index, color_continuous_scale='Blues', text_auto=True, labels=dict(x='Predetto', y='Reale'))
                        st.plotly_chart(cm_fig, use_container_width=True)

                        # ROC & PR interactive
                        if y_score is not None:
                            fpr, tpr, _ = roc_curve(y_true, y_score)
                            auc = roc_auc_score(y_true, y_score)
                            roc_fig = go.Figure()
                            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={auc:.3f}'))
                            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='random'))
                            roc_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', template='plotly_white')
                            st.plotly_chart(roc_fig, use_container_width=True)

                            prec, rec, _ = precision_recall_curve(y_true, y_score)
                            ap = average_precision_score(y_true, y_score)
                            pr_fig = go.Figure()
                            pr_fig.add_trace(go.Scatter(x=rec, y=prec, mode='lines', name=f'AP={ap:.3f}'))
                            pr_fig.update_layout(title='Precision-Recall', xaxis_title='Recall', yaxis_title='Precision', template='plotly_white')
                            st.plotly_chart(pr_fig, use_container_width=True)

                        # Provide downloads
                        csv_bytes = out_df.to_csv(index=False).encode('utf-8')
                        st.download_button('Scarica predizioni (CSV)', data=csv_bytes, file_name='predictions.csv', mime='text/csv')
                        
                        # Download completa tabella con dati paziente + predizioni + label reale
                        st.markdown("---")
                        st.subheader("📥 Download Tabella Completa")
                        st.write("Scarica la tabella completa con i dati dei pazienti, predizioni del modello e valori reali:")
                        
                        # Crea la tabella ordinata per facilità di lettura
                        if y_true is not None:
                            # Ordina le colonne: id, dati paziente, diabetes_true, diabetes_pred, prob_diabetes, is_error
                            cols_order = ["id"] if "id" in out_df.columns else []
                            cols_order += [col for col in out_df.columns if col not in ["id", "diabetes_true", "diabetes_pred", "prob_diabetes", "is_error"]]
                            cols_order += ["diabetes_true", "diabetes_pred"]
                            if "prob_diabetes" in out_df.columns:
                                cols_order.append("prob_diabetes")
                            if "is_error" in out_df.columns:
                                cols_order.append("is_error")
                            
                            out_df_ordered = out_df[[col for col in cols_order if col in out_df.columns]]
                            csv_complete = out_df_ordered.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label='⬇️ Scarica Tabella Completa (CSV)',
                                data=csv_complete,
                                file_name='tabella_completa_predizioni.csv',
                                mime='text/csv',
                                key='download_complete_table'
                            )
                        
                        try:
                            import json

                            st.download_button('Scarica metriche (JSON)', data=json.dumps(metrics, indent=2), file_name='metrics.json', mime='application/json')
                        except Exception:
                            pass
                except Exception as exc:
                    st.error(f"Errore durante l'inferenza: {exc}")

    # Tab: Test & Score
    with tabs[3]:
        st.subheader("Test & Score — Confronto modelli e tecniche")
        st.write("Seleziona modelli e tecniche di validazione da eseguire sul dataset di training. Attenzione: alcune tecniche (tuning) possono richiedere tempo.")
        
        # Mostra risultati salvati in precedenza (se disponibili)
        if st.session_state.test_score_results is not None:
            st.info("✅ **Risultati precedenti disponibili** - I dati del Test & Score sono salvati. Puoi interagire con i grafici senza perdere i risultati!")
        
        if models is None or train_validate is None or preprocessing is None or utils_io is None:
            st.warning("Funzionalità Test & Score non disponibile: moduli mancanti")
        else:
            model_registry = models.get_models()
            model_names = list(model_registry.keys())
            chosen_models = st.multiselect("Modelli da valutare", options=model_names, default=model_names)
            tech_options = st.multiselect("Tecniche", options=["holdout", "cv5", "cv5_tuning"], default=["holdout"]) 
            seed_val = st.number_input("Seed", value=(config.SEED if config else 42), step=1)
            run_button = st.button("Esegui Test & Score")
            if run_button:
                try:
                    train_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"
                    train_df = utils_io.load_csv(train_path)
                except Exception as exc:
                    st.error(f"Impossibile caricare train.csv per eseguire Test & Score: {exc}")
                    train_df = pd.DataFrame()
                if train_df.empty:
                    st.stop()

                numeric_cols, categorical_cols = preprocessing.get_feature_lists(train_df)
                feature_list = numeric_cols + categorical_cols
                if not feature_list:
                    st.error("Nessuna feature valida trovata nel train.csv")
                    st.stop()

                preprocess_pipe = preprocessing.build_preprocess_pipeline(feature_list=feature_list, weight_top_features=False)
                X = train_df.drop(columns=[config.TARGET])
                y = train_df[config.TARGET]

                all_records = []
                detailed_records = {}
                progress = st.progress(0)
                total = max(1, len(chosen_models))
                i = 0
                for model_name in chosen_models:
                    estimator, param_space = model_registry[model_name]
                    st.write(f"Valutando: {model_name}")
                    results = train_validate.evaluate_model_with_techniques(
                        model_name,
                        estimator,
                        preprocess_pipe,
                        param_space,
                        tech_options,
                        X,
                        y,
                        seed=seed_val,
                    )
                    for res in results:
                        rec = {"model": res.model, "technique": res.technique}
                        rec.update(res.metrics)
                        all_records.append(rec)
                        if res.y_true is not None and res.y_pred is not None:
                            key = (res.model, res.technique)
                            detailed_records[key] = {
                                "model": res.model,
                                "technique": res.technique,
                                "y_true": res.y_true.tolist(),
                                "y_pred": res.y_pred.tolist(),
                                "y_score": res.y_score.tolist() if res.y_score is not None else None,
                            }
                    i += 1
                    progress.progress(int(i / total * 100))

                if not all_records:
                    st.info("Nessun risultato prodotto")
                else:
                    results_df = pd.DataFrame(all_records)
                    # Salva i risultati in session_state per evitare perdite con i rerun
                    st.session_state.test_score_results = results_df
                    st.session_state.test_score_models = chosen_models
                    st.session_state.test_score_techniques = tech_options
                    st.session_state.test_score_seed = seed_val
                    st.session_state.test_score_details = detailed_records
                    st.success("✅ Risultati salvati! Puoi ora interagire con i grafici senza perdere i dati.")
            
            # Mostra i risultati salvati (se disponibili)
            if st.session_state.test_score_results is not None:
                results_df = st.session_state.test_score_results
                
                st.subheader("Risultati Test & Score")
                st.dataframe(results_df.sort_values(by=["roc_auc_mean"], ascending=False).reset_index(drop=True))
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button('Scarica risultati (CSV)', data=csv_bytes, file_name='test_and_score_results.csv', mime='text/csv')
                
                # Sezione Grafici di Confronto
                st.subheader("📊 Grafici di Confronto tra Modelli")
                
                # Seleziona metrica da visualizzare
                available_metrics = [col for col in results_df.columns if col not in ["model", "technique"]]
                metric_choice = st.selectbox("Metrica da confrontare", options=available_metrics, index=0 if "roc_auc_mean" in available_metrics else 0)
                
                chart_type = st.radio("Tipo grafico", ["Bar chart (per modello)", "Box plot (per tecnica)", "Scatter (due metriche)"])
                
                if chart_type == "Bar chart (per modello)":
                    # Raggruppa per modello e media la metrica
                    model_avg = results_df.groupby("model")[metric_choice].mean().sort_values(ascending=False)
                    fig_bar = px.bar(
                        x=model_avg.index,
                        y=model_avg.values,
                        labels={"x": "Modello", "y": metric_choice},
                        title=f"Media {metric_choice} per Modello",
                        color=model_avg.values,
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                elif chart_type == "Box plot (per tecnica)":
                    # Box plot per tecnica
                    fig_box = px.box(
                        results_df,
                        x="technique",
                        y=metric_choice,
                        color="technique",
                        title=f"Distribuzione {metric_choice} per Tecnica",
                        points="all"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                elif chart_type == "Scatter (due metriche)":
                    # Scatter plot con due metriche
                    other_metrics = [m for m in available_metrics if m != metric_choice]
                    if other_metrics:
                        metric2 = st.selectbox("Seconda metrica", options=other_metrics)
                        fig_scatter = px.scatter(
                            results_df,
                            x=metric_choice,
                            y=metric2,
                            color="model",
                            symbol="technique",
                            title=f"{metric_choice} vs {metric2}",
                            hover_data={"model": True, "technique": True}
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.info("Seleziona un'altra metrica per il confronto")
                
                st.markdown("---")
                st.subheader("Analisi dettagliata (Holdout)")
                details_map = st.session_state.get("test_score_details", {})
                valid_keys = sorted([key for key, payload in details_map.items() if payload and payload.get("y_true")])
                if not valid_keys:
                    st.info("Curve e matrici sono disponibili solo per i risultati holdout con probabilità salvate.")
                else:
                    selected_key = st.selectbox(
                        "Modello/tecnica da analizzare",
                        options=valid_keys,
                        format_func=lambda x: f"{x[0]} · {x[1]}",
                    )
                    detail_payload = details_map[selected_key]
                    y_true = np.asarray(detail_payload["y_true"])
                    y_pred = np.asarray(detail_payload["y_pred"])
                    y_score = detail_payload["y_score"]
                    y_score = np.asarray(y_score) if y_score is not None else None
                    if evaluation is not None:
                        metric_dict = evaluation.compute_classification_metrics(y_true, y_pred, y_score)
                    else:
                        metric_dict = {
                            "accuracy": accuracy_score(y_true, y_pred),
                            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
                            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
                            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
                            "roc_auc": roc_auc_score(y_true, y_score)
                            if y_score is not None and len(np.unique(y_true)) > 1
                            else None,
                        }
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Accuracy", f"{metric_dict.get('accuracy', 0):.3f}")
                    col_b.metric("F1 macro", f"{metric_dict.get('f1_macro', 0):.3f}")
                    roc_value = metric_dict.get("roc_auc")
                    col_c.metric("ROC-AUC", f"{roc_value:.3f}" if roc_value is not None else "n/a")

                    label_values = sorted(np.unique(np.concatenate([y_true, y_pred])))
                    cm_counts = confusion_matrix(y_true, y_pred, labels=label_values)
                    cm_norm = confusion_matrix(y_true, y_pred, labels=label_values, normalize="true")
                    fig_cm = make_subplots(rows=1, cols=2, subplot_titles=("Counts", "Normalized"))
                    fig_cm.add_trace(
                        go.Heatmap(
                            z=cm_counts,
                            x=label_values,
                            y=label_values,
                            colorscale="Blues",
                            showscale=False,
                            text=cm_counts,
                            texttemplate="%{text}",
                        ),
                        row=1,
                        col=1,
                    )
                    fig_cm.add_trace(
                        go.Heatmap(
                            z=cm_norm,
                            x=label_values,
                            y=label_values,
                            colorscale="Greens",
                            showscale=True,
                            text=np.round(cm_norm, 2),
                            texttemplate="%{text}",
                        ),
                        row=1,
                        col=2,
                    )
                    fig_cm.update_xaxes(title_text="Predetto", row=1, col=1)
                    fig_cm.update_yaxes(title_text="Reale", row=1, col=1)
                    fig_cm.update_layout(height=420, width=900, template="plotly_white")
                    st.plotly_chart(fig_cm, use_container_width=True)

                    if y_score is not None and len(np.unique(y_true)) > 1:
                        fpr, tpr, _ = roc_curve(y_true, y_score)
                        roc_auc = roc_auc_score(y_true, y_score)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(
                            go.Scatter(
                                x=fpr,
                                y=tpr,
                                mode="lines",
                                name=f"AUC={roc_auc:.3f}",
                                line=dict(width=3, color="#1f77b4"),
                                fill="tozeroy",
                                fillcolor="rgba(31,119,180,0.2)",
                            )
                        )
                        fig_roc.add_trace(
                            go.Scatter(
                                x=[0, 1],
                                y=[0, 1],
                                mode="lines",
                                name="Random",
                                line=dict(dash="dash", color="gray"),
                            )
                        )
                        fig_roc.update_layout(
                            title="ROC Curve",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)

                        precision, recall, _ = precision_recall_curve(y_true, y_score)
                        ap = average_precision_score(y_true, y_score)
                        baseline = float(np.mean(y_true))
                        fig_pr = go.Figure()
                        fig_pr.add_trace(
                            go.Scatter(
                                x=recall,
                                y=precision,
                                mode="lines",
                                name=f"AP={ap:.3f}",
                                line=dict(width=3, color="#2ca02c"),
                                fill="tozeroy",
                                fillcolor="rgba(44,160,44,0.2)",
                            )
                        )
                        fig_pr.add_trace(
                            go.Scatter(
                                x=[0, 1],
                                y=[baseline, baseline],
                                mode="lines",
                                name=f"Baseline={baseline:.2f}",
                                line=dict(dash="dash", color="gray"),
                            )
                        )
                        fig_pr.update_layout(
                            title="Precision-Recall",
                            xaxis_title="Recall",
                            yaxis_title="Precision",
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_pr, use_container_width=True)

                # Tabella di dettaglio con opzioni di filtro
                st.subheader("📋 Tabella Dettagliata")
                filter_model = st.multiselect("Filtra per modello", options=results_df["model"].unique(), default=results_df["model"].unique())
                filter_tech = st.multiselect("Filtra per tecnica", options=results_df["technique"].unique(), default=results_df["technique"].unique())
                
                filtered_df = results_df[
                    (results_df["model"].isin(filter_model)) & 
                    (results_df["technique"].isin(filter_tech))
                ].sort_values(by=[metric_choice], ascending=False)
                
                st.dataframe(filtered_df, use_container_width=True)

                # ===== Genera tabella finale con predizioni per ogni modello =====
                st.markdown("---")
                st.subheader("📦 Genera Tabella Finale con Predizioni per Modello")
                st.write(
                    "Questa operazione allena (su tutto il train) ciascun modello selezionato e aggiunge una colonna di predizione per paziente. "
                    "Alla fine la tabella contiene tutte le feature, le colonne di predizione (una per modello) e infine la colonna target reale."
                )

                if 'test_score_models' in st.session_state and st.session_state.test_score_models:
                    default_models_for_final = st.session_state.test_score_models
                else:
                    default_models_for_final = chosen_models

                models_to_run = st.multiselect("Seleziona modelli per generare la tabella finale", options=chosen_models, default=default_models_for_final)
                if st.button("Genera tabella finale con predizioni"):
                    try:
                        train_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"
                        train_df = utils_io.load_csv(train_path)
                    except Exception as exc:
                        st.error(f"Impossibile caricare train.csv per generare la tabella finale: {exc}")
                        train_df = pd.DataFrame()

                    if train_df.empty:
                        st.stop()

                    # Manteniamo ordine e colonne originali (features only)
                    if config and config.TARGET in train_df.columns:
                        feature_df = train_df.drop(columns=[config.TARGET]).copy()
                        true_target = train_df[config.TARGET].copy()
                    else:
                        feature_df = train_df.copy()
                        true_target = None

                    # Prepara preprocessing pipeline se disponibile
                    try:
                        numeric_cols, categorical_cols = preprocessing.get_feature_lists(train_df)
                        feature_list = numeric_cols + categorical_cols
                        preprocess_pipe = preprocessing.build_preprocess_pipeline(feature_list=feature_list, weight_top_features=False)
                    except Exception:
                        preprocess_pipe = None

                    final_df = feature_df.copy()
                    progress_bar = st.progress(0)
                    total = max(1, len(models_to_run))
                    for idx, model_name in enumerate(models_to_run, start=1):
                        progress_bar.progress(int(idx / total * 100))
                        try:
                            estimator, _ = model_registry[model_name]
                            est = clone(estimator)
                            if preprocess_pipe is not None:
                                pipe = Pipeline([('preprocess', deepcopy(preprocess_pipe)), ('est', est)])
                                pipe.fit(feature_df, true_target)
                                preds = pipe.predict(feature_df)
                            else:
                                est.fit(feature_df, true_target)
                                preds = est.predict(feature_df)
                            col_name = f"pred_{model_name}"
                            # assicurati che il nome sia unico
                            if col_name in final_df.columns:
                                col_name = f"pred_{model_name}_{idx}"
                            final_df[col_name] = preds
                        except Exception as exc:
                            st.warning(f"Impossibile generare predizioni per {model_name}: {exc}")

                    # Aggiungi la colonna target reale alla fine
                    if true_target is not None:
                        final_df[config.TARGET] = true_target.values

                    st.session_state.final_prediction_table = final_df
                    st.success("Tabella finale generata e salvata in session state.")

                # Se disponibile, mostra pulsante di download
                if 'final_prediction_table' in st.session_state and st.session_state.final_prediction_table is not None:
                    st.markdown("### 📥 Scarica la tabella finale")
                    csv_final = st.session_state.final_prediction_table.to_csv(index=False).encode('utf-8')
                    st.download_button('⬇️ Scarica Tabella Finale con Predizioni (CSV)', data=csv_final, file_name='tabella_finale_predizioni_per_modello.csv', mime='text/csv')


if __name__ == "__main__":
    main()

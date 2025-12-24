# Cardio Diabetes Project

Pipeline completa per analizzare, addestrare e confrontare modelli di classificazione del diabete usando esclusivamente feature cardiologiche e cliniche. Il flusso copre EDA, preprocessing senza leakage, confronto modelli/tecniche e generazione di report e predizioni finali.

## Requisiti
- Python 3.10+
- Librerie elencate in `requirements.txt`

## Setup rapido
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Struttura principale
```
cardio_diabetes_project/
├─ data/                # posizionare qui train.csv e test.csv
├─ reports/             # output automatico (grafici, tabelle, modelli)
└─ src/                 # codice sorgente
```

## Esecuzione end-to-end
Lancia tutto il flusso (EDA, training, validazioni, report, best model):
```bash
python src/run_all.py \
  --data_train data/train.csv \
  --data_test data/test.csv \
  --techniques holdout,cv5,cv5_tuning \
  --weight_top_features false \
  --seed 42
```
Output attesi:
- `reports/tables/summary_metrics.csv`
- Grafici EDA in `reports/eda/`
- Grafici confronto modelli in `reports/figs/`
- Modello migliore in `reports/models/best_model.joblib`
- Report sintetico `reports/summary.md`

## Inferenza su test.csv
Una volta addestrato il modello migliore:
```bash
python src/infer.py \
  --model reports/models/best_model.joblib \
  --data_test data/test.csv \
  --outfile reports/tables/predictions_test.csv
```
Output: CSV con colonne `id, diabetes_pred, prob_diabetes`.

## Interfaccia grafica (senza drag & drop)
Per un'esperienza simile a Orange (visualizzazione file, dati, grafici, training, test & score, inferenza) è disponibile una GUI Tkinter:
```bash
python -m src.gui_app
```
La finestra offre:
- Editor file per modificare rapidamente `.py`, `.md`, `.csv` ecc.
- Sezione dati per caricare train/test o CSV personalizzati, vedere anteprime e generare grafici (distribuzioni, violin vs target, heatmap).
- Dashboard training per scegliere tecniche (holdout, cv5, cv5_tuning), avviare il flusso completo, monitorare il log e visualizzare la tabella `Test & Score`.
- Scheda inferenza per caricare un modello/joblib, selezionare un file di test dal computer, visualizzare la classifica delle predizioni e scaricare il CSV finale.

## Note metodologiche
- Il target è `diabetes` e non deve essere presente nel test.
- Le feature `glucose`, `id`, `education` vengono sempre escluse per evitare leakage.
- Le feature candidate principali sono: `heartRate`, `BMI`, `sysBP`, `diaBP`, `totChol`, `age`, `sex`, `prevalentHyp`, `BPMeds`, `is_smoking`, `cigsPerDay`, `prevalentStroke`.
- È disponibile l'opzione `--weight_top_features` per enfatizzare leggermente le prime 5 feature.
- Tutte le validazioni usano split e CV stratificati, con seed globale configurabile.

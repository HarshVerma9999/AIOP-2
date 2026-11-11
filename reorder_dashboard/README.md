# Inventory Control Room

Streamlit dashboard for the **AI in Operations, slide 35** project: an AI-enhanced reorder point and inventory optimisation for fresh-grocery store 308 (FreshRetailNet-50K).

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. Run the command from inside this folder, so the dark theme in `.streamlit/config.toml` is picked up.

## Pages

| Page | What it shows |
|---|---|
| Command deck | Headline results and the four policies compared |
| Demand intelligence | Demand hidden by empty shelves, stock-out patterns, forecasting models |
| Safety stock lab | Formula vs Monte Carlo, live |
| Optimiser | Bayesian optimisation, live |
| Scenario lab | How the best decision moves under business conditions |
| Live replay | The final week, re-optimised day by day |
| Replenishment plan | All 73 products, with a CSV download |
| Data integrity | The notebook's 44 checks |
| About | Method and assumptions |

The **control panel** on the left changes the supplier lead time, service level, costs, shelf life and limits. The live pages recompute using the same engine as the notebook. That takes about a second for safety stock, and 3–8 seconds for a new optimisation.

## Files

| File | Purpose |
|---|---|
| `app.py` | The dashboard |
| `engine.py` | Safety stock, Monte Carlo, inventory simulator and Bayesian optimisation (same code as the notebook) |
| `theme.py` | The Midnight Control Room design |
| `data/` | Results exported from the notebook run (same seeds, same numbers) |

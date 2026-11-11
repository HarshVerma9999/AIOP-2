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
| Overview | The headline result for the chosen product: cost, waste, customers served, and a breakdown of why the AI policy wins |
| Demand | Demand hidden by empty shelves, when sales are lost (by hour), forecasting models and the forecast's 80% range |
| Safety stock | Formula vs Monte Carlo (live), what drives the uncertainty, a risk grid, and which products the formula under-protects |
| Optimiser | Bayesian optimisation (live), the cost of every policy, smart vs blind search, and 4 weeks of shelf stock (sawtooth) |
| Scenarios & replay | How the best decision moves under business conditions; the final week replayed day by day |
| Plan | All 73 products, with a CSV download |
| About & checks | Method, glossary, assumptions, and the notebook's 44 data checks |

Only the open page is computed, so pages load quickly. The **control panel** (product, supplier lead time, service level, costs, shelf life,
limits) applies to every page. Safety stock updates in about a second; a new optimisation takes a few seconds.

## Files

| File | Purpose |
|---|---|
| `app.py` | The dashboard |
| `engine.py` | Safety stock, Monte Carlo, inventory simulator and Bayesian optimisation (same code as the notebook) |
| `theme.py` | The Midnight Control Room design |
| `data/` | Results exported from the notebook run (same seeds, same numbers) |

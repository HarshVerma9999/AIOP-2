# AI in Operations (AIOP): Lab Projects

**Harsh Verma (as25dxb024)**: AI in Operations, November 2026

| Folder | Project |
|---|---|
| [`day1_number_plate_recognition/`](day1_number_plate_recognition) | Day 1 lab: detect and read vehicle number plates from an image. YOLO11 detector + fast-plate-ocr reader. |
| [`day2_reorder_point_optimisation/`](day2_reorder_point_optimisation) | Day 2 exercise (slide 35): AI-enhanced reorder point. ML demand forecasting, dynamic safety stock (formula + Monte Carlo) and Bayesian optimisation under constraints. |
| [`reorder_dashboard/`](reorder_dashboard) | **Inventory Control Room**: a Streamlit dashboard for the Day 2 project, with live what-if controls. |

## Day 2 at a glance

On fresh-grocery store 308 (73 products, FreshRetailNet-50K), tested on 300 random future scenarios:

| Policy | Cost / day | Demand served | Waste | Within all limits? |
|---|---|---|---|---|
| Class calculator | 0.503 | 97.3% | 5.2% | No |
| Textbook formula + EOQ (capped) | 0.757 | 94.4% | 27.4% | No |
| **AI: Bayesian-optimised** | **0.459** | **95.5%** | **3.9%** | **Yes: the only one** |

## Run the dashboard

```bash
cd reorder_dashboard
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501.

**Deploy on Streamlit Community Cloud:**
1. Choose *Create app* → this repository → branch `main`.
2. Set the main file path to `reorder_dashboard/app.py`.

## Run the notebooks

Open either `.ipynb` in Google Colab and choose *Runtime → Run all*. Both notebooks download their data automatically.

| Notebook | Settings | Time |
|---|---|---|
| Day 1 | T4 GPU | about 35–45 min |
| Day 2 | CPU | about 10–15 min |

## Data and licences

| Item | Source and licence |
|---|---|
| Day 1 data | [Vehicle Registration Plates](https://huggingface.co/datasets/keremberke/license-plate-object-detection) (Augmented Startups via Roboflow, CC BY 4.0) |
| Day 2 data | [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (Dingdong-Inc, CC BY 4.0) |
| Ultralytics YOLO | AGPL-3.0 |
| fast-plate-ocr | MIT |
| Optuna | MIT |

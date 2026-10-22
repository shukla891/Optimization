# Chennai Dark Store Location Optimizer

A Streamlit app for exploring candidate dark-store placement in Chennai. It generates reproducible synthetic customer demand and candidate sites, enforces a customer-to-store maximum distance, estimates daily net profit, compares store budgets of 2, 5, 10, 20, and 25, and maps candidate sites and a sample of customer assignments.

## Run locally

Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app defaults to 50 candidate sites, 10,000 customers, a 5 km maximum distance, and a fixed random seed. Adjust candidate count, customer count, distance limit, unit economics, and seed in the sidebar. The map uses OpenStreetMap tiles and synthetic points around Chennai neighborhoods.

## Model

The sidebar lets users set the maximum stores allowed to open and adjust revenue per order, variable cost per order, delivery cost per order per km, daily operating cost per store, setup cost per store, and the setup-cost recovery period. The store-budget comparison includes the standard budgets up to the selected maximum, plus the selected maximum if it is not a standard budget. Defaults are 25 maximum stores, Rs 500 revenue, Rs 445 variable costs, Rs 5 delivery cost per km, Rs 5,000 daily operating costs, Rs 600,000 setup cost, and a three-year recovery period. These are illustrative assumptions, not market-validated figures.

The app derives contribution before delivery as `revenue per order - variable cost per order`. For each customer, eligible daily contribution is expected orders per day multiplied by `max(contribution before delivery - delivery cost per km × straight-line distance, 0)`, subject to the maximum service distance. Estimated net profit per day subtracts each opened store's daily operating cost and its setup cost divided evenly across the selected recovery period (365 days per year; no interest or inflation). The optimizer accounts for those per-store costs when choosing locations.

Demand and candidate locations are synthetic, not observed business data. Customer order rates are randomly generated and constrained to 0.15–4 orders per day. Results do not model taxes, competition, or demand changes in response to price or store placement, so they are not a financial forecast. Candidate sites are selected with a greedy marginal-profit heuristic. This is efficient for interactive use but does not guarantee a globally optimal facility-location solution. Straight-line distance is used instead of road-network or travel distance.

## Deployment

The app is a standard Streamlit entrypoint (`app.py`) with dependencies listed in `requirements.txt`. The requested `share.streamline.io` host did not resolve from this environment, and this workspace has no connected remote repository or hosting credentials, so a public deployment link could not be created here. Once a Streamlit-compatible host and repository access are available, configure it to install `requirements.txt` and run `streamlit run app.py`.

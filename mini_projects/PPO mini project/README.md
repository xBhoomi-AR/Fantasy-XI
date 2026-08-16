# PPO Portfolio Management

Project layout:

- `data/raw/`: downloaded price data.
- `data/processed/`: engineered feature data.
- `src/`: reusable data, environment, and training code.
- `results/models/`: saved PPO policies.
- `results/figures/`: evaluation charts.

Install dependencies with `pip install -r requirements.txt`, then run:

```powershell
python src/train.py
python src/evaluate.py
```

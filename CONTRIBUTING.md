# Contributing to KitchenPulse

Thank you for your interest in improving KitchenPulse! This guide outlines how to set up the project locally, run tests, and contribute changes.

---

## Getting Started

### Prerequisites
- Python 3.10 or later
- Git
- pip (Python package manager)

### Local Setup

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd KitchenPulse
```

#### 2. Create a Virtual Environment
```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Install Development Tools (Optional)
For running tests and linting:
```bash
pip install pytest black pylint
```

---

## Running the Project

### Phase 1: Generate Synthetic Data
Creates a realistic dataset of 17,594 orders across 50 restaurants with merchant bias and hidden load patterns.

```bash
python data/generate_synthetic_data.py
```

**Output:** `data/synthetic_orders.csv`

### Phase 2: Run Simulation
Compares three strategies (baseline, de-biased, full KitchenPulse) head-to-head.

```bash
python simulation/run_simulation.py
```

**Output:**
- `report/figures/simulation_results.png` (5-panel comparison chart)
- `data/processed_orders.csv` (enriched dataset)
- Console output: metrics summary by restaurant tier

### Phase 3: Generate Analysis Charts
Produces publication-quality visualizations of signal correlations, improvements by tier, and more.

```bash
python analysis/correlation_analysis.py
```

**Output:** 6 charts in `report/figures/`:
- `chart1_correlation_heatmap.png` — Signal correlation matrix
- `chart3_hidden_load_impact.png` — Proof of hidden load effect
- `chart4_hourly_kli_heatmap.png` — Kitchen load intensity over time
- `chart5_tier_improvement.png` — Results by restaurant tier
- `chart6_signal_accuracy_ladder.png` — Before/after signal ranking
- `simulation_results.png` — Full 5-panel comparison

### Phase 4: Run Robustness Tests
Validates system resilience against data corruption, missing signals, and out-of-distribution loads.

```bash
python analysis/robustness_tests.py
```

**Output:**
- `ablation_results.png` — MAE when each signal is dropped
- `ood_stress_test.png` — MAE under varying hidden load levels
- Console: bootstrap confidence intervals and statistical significance tests

---

## Project Structure

```
KitchenPulse-Zomato-KPT/
├── README.md                    # Quick start & overview
├── requirements.txt             # Python dependencies
│
├── docs/                        # Documentation
│   ├── DETAILS.md              # Extended problem & algorithms
│   ├── ARCHITECTURE.md         # Production deployment guide
│   ├── README_SUMMARY.md       # 1-page executive summary
│   └── CONTRIBUTING.md         # This file
│
├── data/                        # Datasets & generation
│   ├── generate_synthetic_data.py
│   ├── synthetic_orders.csv
│   ├── processed_orders.csv
│   └── restaurants.csv
│
├── pipeline/                    # Core business logic
│   ├── signal_denoiser.py      # FOR bias detection
│   ├── kitchen_load_index.py   # KLI computation
│   └── feature_store_builder.py # Feature storage reference
│
├── simulation/                  # Validation framework
│   └── run_simulation.py        # 3-way strategy comparison
│
├── analysis/                    # Charts & statistical tests
│   ├── correlation_analysis.py  # Signal analysis
│   └── robustness_tests.py      # Failure mode testing
│
└── report/
    └── figures/                 # Generated visualizations
```

---

## Code Style & Testing

### Code Standards
- **Python:** PEP 8 with 100-character line limit
- **Imports:** Organize as std lib, third-party, local
- **Docstrings:** Google-style docstrings for functions and classes
- **Type hints:** Use where practical (Python 3.10+)

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_signal_denoiser.py

# Run with coverage
pytest --cov=pipeline --cov=analysis
```

### Linting
```bash
# Check code style
pylint pipeline/ analysis/ simulation/

# Auto-format code
black pipeline/ analysis/ simulation/
```

---

## Making Changes

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Edit files in their respective modules
- Update `requirements.txt` if adding dependencies
- Add tests for new functionality

### 3. Run the Full Pipeline
```bash
python data/generate_synthetic_data.py
python simulation/run_simulation.py
python analysis/correlation_analysis.py
python analysis/robustness_tests.py
```

Verify output looks correct and metrics improve or stay stable.

### 4. Commit with Clear Messages
```bash
git add .
git commit -m "feat: add new KLI signal for ambient temperature"
# or
git commit -m "fix: handle missing foot traffic data gracefully"
# or
git commit -m "docs: update architecture with deployment guide"
```

Use conventional commit prefixes:
- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation
- `refactor:` — Code cleanup
- `test:` — Test additions
- `chore:` — Dependencies, tooling

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a PR on GitHub with:
- **Title:** Clear, one-line summary
- **Description:** What changed, why, and any caveats
- **Testing:** Steps to validate the change
- **Links:** Reference related issues

---

## Code of Conduct

- Be respectful and inclusive
- Assume good intent in discussions
- Focus on ideas, not individuals
- Report abuse to the project maintainers

---

## Architecture Overview

### Signal Denoiser (pipeline/signal_denoiser.py)
Removes merchant bias from "Food Ready" button timestamps:
- `flag_rider_proximate()` — Detects suspicious timing
- `compute_bias_offsets()` — Learns per-merchant bias
- `apply_for_correction()` — Applies denoising

### Kitchen Load Index (pipeline/kitchen_load_index.py)
Computes real-time load score from 4 signals:
- `normalise_*()` — Component normalization
- `compute_kli()` — Weighted fusion
- `apply_kli_to_kpt()` — Final adjustment

### Simulation (simulation/run_simulation.py)
3-way strategy comparison:
- **Baseline:** partner platform's current model
- **De-biased:** FOR correction only
- **KitchenPulse:** Full pipeline
- Output: KPT MAE, rider wait, tier breakdown

### Analysis (analysis/correlation_analysis.py, robustness_tests.py)
Statistical validation:
- Signal correlations with true KPT
- Improvement by restaurant tier
- Robustness under 6 failure modes
- Bootstrap confidence intervals

---

## Debugging Tips

### Check Signal Quality
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/processed_orders.csv')
print('KLI correlation with true KPT:', df[['kli_score', 'true_kpt_minutes']].corr())
"
```

### Trace Pipeline Steps
Edit `simulation/run_simulation.py` and add print statements:
```python
print(f"DEBUG: KLI score = {kli_score}, DeMarcation = {correction_factor}")
```

### Validate Data Generation
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/synthetic_orders.csv')
print('Biased merchants:', (df['honest_merchant'] == False).sum())
print('Avg merchant bias:', df[df['honest_merchant'] == False]['merchant_delay_minutes'].mean())
"
```

---

## Frequently Asked Questions

**Q: How do I update the foot traffic signal mapping?**  
A: Edit the `normalise_foot_traffic()` function in `kitchen_load_index.py`. The function expects a `foot_traffic_index` column (0–100 scale).

**Q: What if a restaurant has no historical data?**  
A: The denoiser falls back to default bias (0). The KLI redistributes weights to available signals. See docs/ARCHITECTURE.md for fallback details.

**Q: Can I use real data instead of synthetic data?**  
A: Yes. Replace `synthetic_orders.csv` with your real data, ensuring it has these columns:
- `restaurant_id`, `order_id`, `timestamp`
- `for_button_time`, `actual_ready_time`
- `concurrent_platform_orders`, `foot_traffic_index`
- And all other signals expected by the pipeline

**Q: How do I add a new signal to KLI?**  
A: 1. Add a normalize function in `kitchen_load_index.py`  
2. Add a weight in the `compute_kli()` function  
3. Update the fallback logic if the signal can be unavailable  
4. Re-run simulation to measure impact

---

## Reporting Issues

Found a bug? Unexpected behavior? Please open an issue on GitHub with:
1. **Description:** What went wrong?
2. **Steps to reproduce:** How to trigger the issue?
3. **Expected vs. actual:** What should happen?
4. **Environment:** Python version, OS, dependencies version

---

## Performance Notes

- **Data generation:** ~5 seconds for 17.5K orders
- **Simulation:** ~20 seconds (single-threaded)
- **Analysis charts:** ~30 seconds
- **Robustness tests:** ~60 seconds (includes 1000 bootstrap resamples)

If runs are slow, check:
- Disk I/O (use local SSD, not network drive)
- RAM usage (system has > 8 GB free?)
- Background processes (close unnecessary apps)

---

## Resources

- **Problem domain:** See docs/DETAILS.md for extended background
- **Production architecture:** See docs/ARCHITECTURE.md for deployment guide
- **Executive summary:** See docs/README_SUMMARY.md for judges/stakeholders

---

## Contact

Questions or suggestions? Open an issue or reach out to the maintainers.

**Happy contributing!** 🚀


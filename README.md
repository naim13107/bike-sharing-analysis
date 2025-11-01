# Bike Sharing Demand Prediction: Integrated Machine Learning and Signal Processing Approach

## Abstract
This research presents a comprehensive framework combining machine learning with signal processing techniques for bike-sharing demand prediction. The study employs advanced feature engineering, Fourier analysis for periodicity detection, and adapted signal processing metrics (PSNR/SNR) for regression evaluation.

## Research Contributions
- **Novel Feature Engineering**: Temporal dependencies and weather interactions
- **Signal Processing Integration**: Fourier analysis revealing dominant periodicities
- **Interpretable AI**: SHAP analysis for model transparency
- **Urban Planning Applications**: Data-driven policy recommendations

## Key Results
- **Best Model: Gradient Boosting (R² = 0.8706)
- **Dominant Features: Rolling 7-day mean (46.64%), comfort index (11.27%)
- **Periodicities Identified: 365.5 days (annual), 243.7 days (seasonal)

## Reproduction
```bash
git clone https://github.com/naim13107/bike-sharing-analysis.git
cd bike-sharing-analysis
pip install -r requirements.txt
python bike2.py


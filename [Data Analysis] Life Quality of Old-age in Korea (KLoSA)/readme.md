# Life Quality of Old-age in Korea

A data analysis project quantifying and exploring quality of life among Korea’s elderly population. It integrates subjective satisfaction measures with objective health, economic, and social indicators to derive both individual and aggregate life quality scores.

---

## 📋 Overview

This project aims to:

1. **Measure Individual Life Quality**: Calculate a composite Life Quality Score by integrating subjective satisfaction and objective health, economic, and social indicators.
2. **Assess Aggregate Quality**: Present an Adjusted Total Life Quality Score that accounts for generational disparities, revealing true trends over time.

---

## 🧰 Technology Stack

- **Language & Environment**: Python 3.8+ in Jupyter Notebook
- **Data Manipulation**: pandas, numpy
- **Statistical Modeling**: statsmodels, scikit-learn, scipy
- **Explainability**: SHAP for factor importance visualization
- **Visualization**: matplotlib, seaborn, Plotly
- **Documentation**: Markdown

---

## 📂 Project Structure

```plaintext
├── data/
├── population.ipynb            # Descriptive EDA: satisfaction, demographics, temporal trends
├── factor_analysis.ipynb       # Regression, ANOVA, SHAP analysis of key life-quality factors
├── Life Quality of Old-age in Korea.pdf   # Final project report summarizing methods & findings
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 📈 Data Source

- **Dataset**: Korean Longitudinal Study of Aging (KLoSA) 2nd wave (2008) through 9th wave (2022)
- **Publisher**: Korea Employment Information Service
- **Sample**: \~10,254 individuals aged 45+ across multiple biennial waves

---

## 🚀 Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/youruser/life-quality-elderly-kr.git
   cd life-quality-elderly-kr
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Obtain KLoSA data**

   - Place the CSV file (`klosa_panel_data.csv`) into the `data/` directory.

4. **Run Notebooks**

   - Launch Jupyter Lab or Notebook:
     ```bash
     jupyter lab
     ```
   - Execute `population.ipynb` for descriptive analysis.
   - Execute `factor_analysis.ipynb` for modeling and explainability.

---

## 🛠 Methodology

1. **Descriptive Analysis** (`population.ipynb`):

   - Summarize subjective life satisfaction rates and objective health/economic indicators by age group and year.
   - Visualize trends and cross-sectional distributions.

2. **Composite Scoring**:

   - Combine normalized subjective and objective subfields (health, financial status, social participation) into a single Life Quality Score using weighted aggregation and inequality measures.

3. **Aggregate Adjustment**:

   - Compute an Adjusted Total Life Quality Score by controlling for birth-cohort effects to reveal true intergenerational trends.

4. **Factor Analysis** (`factor_analysis.ipynb`):

   - Use linear regression and ANOVA to identify variables contributing most to score variance (age, birth year, income, health indices).
   - Apply SHAP explanations on regression and random forest models to quantify factor importance and interaction effects.

5. **Visualization**:

   - Histograms of life quality score distribution.
   - Time-series and cohort plots of score evolution.
   - Factor importance bar charts and SHAP summary plots.

---

## 🔧 Customization

- Adjust weighting schemes for subjective vs. objective components in `factor_analysis.ipynb`.
- Incorporate additional subfields (e.g., mental health scales) by extending the data preprocessing pipeline.
- Experiment with alternative inequality measures (e.g., Gini coefficient) in composite scoring.

---

## 📜 License

This project is licensed under the MIT License.

*Last updated: 2025-07-05*


# UK Retail CRM Analytics

A comprehensive notebook-based analysis of UK online retail transactional data, designed to derive actionable insights for customer relationship management (CRM).

---

## 📋 Overview

This project analyzes invoice-level data from a UK-based online retailer to understand purchasing behaviors and inform CRM strategies. Key goals include:

1. **Data Exploration & Cleaning**: Audit data quality, handle missing or duplicate records, and prepare a reliable dataset.
2. **Feature Engineering**: Create metrics such as order frequency, monetary value, and product basket compositions.
3. **Statistical & Cluster Analysis**: Segment customers using RFM, K-Means, DBSCAN, and k-Modes for categorical patterns.
4. **Cohort & Retention Analysis**: Track customer retention over time and evaluate segment-specific drop-off rates.
5. **Visual Storytelling**: Generate compelling visualizations (bar charts, radar plots, world maps) to communicate insights.
6. **Recommendations**: Provide targeted marketing and operational suggestions based on analytical findings.

---

## 🧰 Technology Stack

- **Language & Platform**: Python 3.8+ in Jupyter Notebook
- **Data Handling**: pandas, Polars, numpy
- **Statistical Modeling**: scikit-learn (KMeans, DBSCAN, PCA), kmodes
- **Text Processing**: nltk
- **Visualization**: matplotlib, seaborn, Plotly (plotly.graph\_objs)
- **Clustering Utilities**: silhouette score, pairwise distances
- **Utilities**: scipy, tqdm

---

## 📂 Project Structure

```plaintext
├── data/
│   └── online_retail.csv         # Raw transactional data from UCI Repository
├── crm_analytics.ipynb       # End-to-end analysis and visualizations
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 📈 Data Source

- **Dataset**: Online Retail Data Set (UCI Machine Learning Repository)
- **Period**: December 1, 2010 – December 9, 2011
- **Records**: Invoices, Stock Codes, Descriptions, Quantities, Unit Prices, Customer IDs, Invoice Dates, Countries
- **Access**: Download from [https://archive.ics.uci.edu/ml/datasets/Online+Retail](https://archive.ics.uci.edu/ml/datasets/Online+Retail)

---

## 🚀 Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/youruser/uk-retail-crm-analytics.git
   cd uk-retail-crm-analytics
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Obtain data**

   - Download `Online Retail.csv` from the UCI repository and place it in the `data/` directory as `online_retail.csv`.

4. **Launch analysis notebook**

   ```bash
   jupyter lab
   ```

   - Open `notebooks/crm_analytics.ipynb` and run cells sequentially.

---

## 🛠 Methodology

1. **Data Preparation**

   - Inspect schema, identify missing CustomerID entries, drop or impute as appropriate.
   - Remove cancellations and correct negative quantities.
   - Deduplicate repeated lines and standardize product descriptions.

2. **Exploratory Data Analysis**

   - Summarize sales volume, revenue, product diversity, and geographic distribution (world map of orders).
   - Analyze basket size and order frequency distributions.

3. **Product Category Insights**

   - Group SKUs into categories, visualize top-selling segments, and seasonal patterns.

4. **Customer Segmentation**

   - **Category-Based Clustering**: Apply k-Modes to identify groups by preferred product categories.
   - **RFM Analysis**: Compute Recency, Frequency, Monetary metrics; segment customers and profile behavior.
   - **Spatial Clustering**: Use DBSCAN to detect geographic customer clusters.

5. **Cohort & Retention Analysis**

   - Construct cohort tables by first purchase month; visualize retention heatmap.
   - Compare retention rates across RFM or category-based segments; derive insights on customer loyalty.

6. **Recommendations**

   - Target high-value but at-risk customers with tailored promotions.
   - Design loyalty programs for segments with low retention.
   - Optimize inventory for top categories and cross-sell complementary products.

---

## 🔧 Customization

- Adjust clustering parameters (number of clusters, eps) for segmentation.
- Extend feature set with time-based behaviors (e.g., time of day, weekday patterns).
- Integrate external data (e.g., demographic or marketing campaign data) for enriched analysis.

---

## 📜 License

This project is licensed under the MIT License.

*Last updated: 2025-07-05*


# 🚚 Intelligent Supply Chain & Inventory Optimizer

An end-to-end AI & Data Engineering solution integrating heterogeneous relational and non-relational database topologies to proactively predict logistics disruptions, optimize inventory buffers, and monitor fleet operations in real time.

---

## 📌 Core Architecture & Tech Stack

- **Backend Framework:** Django (Python)
- **Hybrid Persistence Layer:**
  - **PostgreSQL:** Relational/transactional layer for canonical order records, SKUs, and ACID compliance.
  - **MongoDB:** Contextual/document store for high-frequency IoT telemetry, weather arrays, and driver incident logs.
- **Analytical Warehouse:** Integrated columnar Parquet warehouse (`integrated_data_warehouse.parquet`) joined via canonical `Asset_ID`.
- **Machine Learning & Mining Engines:**
  - **Random Forest Classifier (`scikit-learn`):** Multi-class delay and order risk prediction (~99.9% validation baseline).
  - **FP-Growth Mining (`mlxtend`):** Tree-based association rule mining mapping environmental conditions to latency risk.
  - **Hybrid NLP Engine:** Driver sentiment parsing, stress level quantification, and automated safety dispatch prompts.
- **Frontend / Visualization:** Dynamic Plotly, Seaborn, HTML5, Bootstrap templates.

---

## 🚀 Key Features

1. **Reactive Fleet Monitor:** Dual-stream telemetry visualization combining static warehouse batch trends with live streaming ingestion dots.
2. **Predictive Latency Scoring:** Automated delay risk assessment based on scheduled vs. real shipping transit cycles.
3. **Driver Stress & Sentiment Analysis:** Natural language evaluation of field logs for proactive fleet welfare and safety alerts.
4. **Quick Real-Time Entry Ingest:** Sub-second transactional ingestion form synchronizing database updates with downstream analytical assets.

---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Alishba222/intelligent-supply-chain-optimizer.git](https://github.com/Alishba222/intelligent-supply-chain-optimizer.git)
cd intelligent-supply-chain-optimizer

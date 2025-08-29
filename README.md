# Results vs Broker Expectations (Streamlit)

Compare **Actual** vs **Brokers' Expected** (Sales/EBITDA/PAT), and view **Beat/Inline/Miss** distribution.
- Login via Streamlit secrets
- CSV upload with schema validation
- Grouped-by-metric chart (Actual + Expected per broker)
- Beat/Inline/Miss% chart
- Tables and CSV download

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py

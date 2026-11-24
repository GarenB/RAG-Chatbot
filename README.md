# Run Project

### 1. Start Docker Services

```bash
docker compose up -d
```

### 2. Start FastAPI Backend

```bash
uvicorn src.main:app --reload
```

### 3. Start Streamlit Frontend

```bash
uv run streamlit run src/streamlit_app.py
```

### 4. Stop Docker Services

```bash
docker compose down -v
```

---

# Run Evaluation

Execute the RAGAS evaluation:

```bash
python -m src.evaluation.evaluate
```

### Evaluation Results

```text
{
    'context_recall': 1.0000,
    'faithfulness': 0.7667,
    'factual_correctness(mode=f1)': 0.6740
}
```

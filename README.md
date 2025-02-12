# StructFact: Benchmarking Structured Factual Reasoning in Large Language Models

## Introduction
Large Language Models (LLMs) demonstrate remarkable capabilities in NLP tasks but face significant challenges when reasoning over structured factual knowledge. Structured data introduces unique characteristics that impact LLM performance:

1. **Heterogeneity** - Mixed data types (text, numbers, dates) 
2. **Topological Interdependencies** - Complex structural relationships
3. **Order Invariance** - Permutation-invariant semantics
4. **Sparsity** - Handling missing values
5. **Lack of Prior Knowledge** - Domain-specific context sensitivity

To address these challenges, we present **StructFact** - a comprehensive benchmark with:
- 📊 **13,407 factual queries** across diverse structures (tables/lists/graphs)
- 🌍 Multi-domain coverage with temporal/regional variations
- 🧩 5 reasoning tasks: Arithmetic Calculation, Geography-Time Reasoning, Multi-hop Reasoning, Composition Understanding, and Combining Structural and Unstructural Reasoning
- 🆕 **StructFact-Unseen** subset for testing generalization on fresh knowledge

### File Structure
```
├── data/
│   └── dataset_demo.json    # Sample dataset entries
├── src/
│   ├── cal_option.py        # Metric calculation script
│   └── run_llm.py          # Model inference script
```

### Usage
1. **Run Inference**  
   Configure your LLM in `run_llm.sh`:

   Then execute:
   ```bash
   chmod +x run_llm.sh
   ./run_llm.sh
   ```

2. **Calculate Metrics**  `
   Generate accuracy and task-specific metrics:
   ```bash
   python src/cal_option.py /path/to/your_llm_output
   ```

# amazing-organized-data
Computational Biology datasets and Simulated datasets for causal inference with multiple treatments.

## Usage:

**Copula simulated dataset**: Gaussian nonlinear 
```python
sdata_copula = copula_simulated_data()
X, y, y01,treatement_columns ,treatment_effects = sdata_copula.generate_samples()
```

**GWAS simulated dataset**: Sparse Effects 
```python
sdata_gwas = gwas_simulated_data()
X, y, y01, treatement_columns, treatment_effects  = sdata_gwas.generate_samples()
```

## References:
- https://github.com/JiajingZ/CopulaSensitivity
- https://github.com/raquelaoki/ParKCa

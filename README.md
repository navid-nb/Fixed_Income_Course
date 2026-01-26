# Fixed-Income Securities Project (60201)

This project contains tools for yield curve smoothing and modeling, specifically implementing the Nelson-Siegel-Svensson (NSS) model.

## Project Structure
- `src/`: Core Python logic and mathematical functions (e.g., NSS model).
- `notebooks/`: Jupyter notebooks.
- `requirements.txt`: List of necessary Python libraries.

---

## Setup Instructions
 

Create a virtual python environment `venv` and activate it:
```bash
python -m venv .venv
.venv\Scripts\activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

Run `main.ipynb`. It will detect that no prior data exists 
and will pull it from WRDS. Any subsequent run of `main.ipynb` 
will reuse the last pulled data.
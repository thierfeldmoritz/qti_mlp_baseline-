# QTI_ML environment setup (Windows PowerShell)
# Run from repo root: C:\QTI_ML

# 1) Create environment from conda file (recommended)
conda env create -f .\environment.yml

# 2) Activate env
conda activate QTI_Project

# 3) Register kernel for notebooks in VS Code
python -m ipykernel install --user --name QTI_Project --display-name "Python (QTI_Project)"

# 4) Optional: If you want pip-only install in an existing env instead
# python -m pip install --upgrade pip
# python -m pip install -r .\requirements.txt

# 5) Quick dependency smoke-test
python .\verify_environment.py

# Imaging mass cytometry (IMC) profiling of a case of Metastatic Sarcomatoid Urothelial Carcinoma

Tumor-Immune Microenvironment Revealed by Imaging Mass Cytometry in a Metastatic Sarcomatoid Urothelial Carcinoma with a Prolonged Response to Pembrolizumab

Hussein Alnajar\*, Hiranmayi Ravichandran\*, André F. Rendeiro\*, Kentaro Ohara, Wael Al Zoughbi, Jyothi Manohar, Noah Greco, Michael Sigouros, Jesse Fox, Emily Muth, Samuel Angiuoli, Bishoy Faltas, Michael Shusterman, Cora N. Sternberg, Olivier Elemento, Juan Miguel Mosquera. Cold Spring Harbor Molecular Case Studies. 2022. doi:

Imaging mass cytometry data is available from the following Zenodo repository: https://doi.org/10.5281/zenodo.6251221

## Reproducibility

### Requirements

- Python 3.7+ (running on 3.8.2)
- Python packages as specified in the [requirements file](requirements.txt) - install with `make requirements` or `pip install -r requirements.txt`.

### Download data

Please download files available in the Zenodo repository and place them in a folder named `processed/20200122_PD_L1_100_percent_case/tiffs` from the root of the project (e.g. if cloning the repository).

### Running

Essentially all analysis is [in one file](src/case.PM2078.analysis.py), which can be run for example like:
```
python src/case.PM2078.analysis.py
```

Additional steps that were used to process the raw data from a MCD file can be seen in the [Makefile](Makefile).

This is roughly equivalent to running `imc process ${TIFF_FILES}`.

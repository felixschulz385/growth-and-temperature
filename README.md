# 🌍 Growth and Temperature (GNT) Data System

A satellite data processing system for studying **the direct impact of economic growth on local temperature**, investigating how economic development affects local warming independent of global CO₂ effects.

## 🎯 Research Question

**How much does local economic growth contribute to local warming, independent of global CO₂ effects?**

Economic growth may directly alter local temperatures through:
- **Land Cover Change**: Deforestation, urbanization, irrigation
- **Aerosol Pollution**: Industrial particles affecting albedo  
- **Anthropogenic Heat**: Direct thermal emissions from economic activity

## 🔬 Research Innovation

This project goes beyond existing urban heat island studies:

- **Global scope**: Entire planet, not just selected cities
- **Growth dynamics**: Economic change over time, not static comparisons
- **Rural inclusion**: All development effects, not just urban areas
- **Causal design**: Natural experiments vs. correlational evidence
- **High resolution**: 30+ years of satellite data at 500m grid resolution

## 📊 Data & Methodology

### Core Model
Two-way fixed-effects panel regression:
```
T_it = α + β · NightLights_it + γ_i + δ_t + λ_i · t + ε_it
```

### Data Sources
- **Economic Activity**: DMSP-OLS (1992–2013), VIIRS-DNB (2012–2022) nighttime lights
- **Temperature**: AVHRR & MODIS LST from GLASS archive
- **Supporting**: ESA CCI land cover, administrative boundaries

### Sample Scale
- **Spatial**: Global 500m × 500m grid cells
- **Temporal**: 1992–2022 (18+ billion observations)
- **Causal ID**: Regional favoritism, resource discoveries

## 🚀 Quick Start

### Installation
```bash
git clone <repository-url>
cd growth-and-temperature
conda env create -f environment.yml
conda activate gnt
pip install -e .
```

### Basic Usage
```bash
# Download data
python run.py download --config orchestration/configs/data.yaml --source glass

# Process data
python run.py preprocess --config orchestration/configs/data.yaml --source glass_modis --stage annual

# Assemble datasets
python gnt/data/assemble/0_main.py
```

### HPC Processing
```bash
# Submit SLURM jobs for large-scale processing
sbatch orchestration/slurm/glass-modis-preprocess-annual.sh
sbatch orchestration/slurm/eog-dvnl-preprocess-tabular.sh
sbatch orchestration/slurm/assemble_0_main.sh
```

## 🏗️ System Architecture

### Processing Pipeline
1. **Download**: Multi-source data acquisition with retry logic
2. **Preprocess**: Temporal aggregation (Daily → Annual), spatial harmonization
3. **Assemble**: Analysis-ready datasets with consistent alignment

### Key Features
- **Unified Interface**: Single `run.py` script for all operations
- **SLURM Integration**: Pre-configured HPC job scripts
- **Scalable Processing**: Dask-based parallel processing
- **Data Standards**: Chunked Zarr format for efficient I/O

## 📁 Repository Structure
```
.
├── gnt/                     # Core Python package
│   ├── data/               # Data processing modules
│   └── experiments/        # Analysis notebooks
├── orchestration/          # Configuration & SLURM scripts
├── scripts/               # Utility scripts
└── data_nobackup/         # Processed data (not in git)
```

## 🎯 Research Applications

### Current Focus
- Urban heat island quantification
- Economic development impact assessment
- Regional climate pattern analysis

### Policy Relevance
If causal effects confirmed:
- Climate cost accounting for development projects
- Urban planning and industrial zoning optimization
- Welfare impact studies linking temperature to human outcomes

## 📅 Project Status
- ✅ **Completed**: Data harmonization, pilot results
- 🔄 **In Progress**: Full-scale global estimation  
- 📋 **Next**: Welfare impacts, mechanism analysis

## 📞 Contact
**Felix Schulz** - felix.schulz@unibas.ch

## 🔗 Resources
- [GLASS Data Portal](https://glass.hku.hk/)
- [EOG Nighttime Lights](https://eogdata.mines.edu/nighttime_light/)
- [ESA Climate Change Initiative](https://climate.esa.int/)

## 📄 License
MIT License - see [LICENSE](LICENSE) file for details.
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
conda activate src
pip install -e .
```

### Basic Usage
```bash
# Fetch + prepare a source (fetch/prepare/grid lifecycle)
python run.py data run --config orchestration/configs/data.yaml --source glass_modis --step prepare

# Assemble the panel: every source in assembly.sources, on the chosen grid.
# --grid picks the output resolution; --shake adds grid-origin robustness variants.
python run.py assemble create --config orchestration/configs/data.yaml --grid 1km
python run.py assemble create --config orchestration/configs/data.yaml --grid 10km --shake quad
# output: ${DATA_NOBACKUP}/assembled/grid=<label>/shake=<base|s0|s1|...>/ix=/iy=/data.parquet

# Refresh one source in an already-built table
python run.py assemble update --config orchestration/configs/data.yaml --grid 10km --datasource eog_viirs
```

### HPC Processing
```bash
# Submit a single (source, step) as a SLURM job (resource defaults from
# orchestration/configs/slurm_jobs.yaml, overridable with --slurm-time/-mem/-cpus/-qos/-partition)
python run.py data run --source glass_modis --step prepare --slurm

# Submit a source's full dependency chain (REQUIRES prerequisites included)
python run.py data run --source glass_modis --step prepare --slurm --chain

# Preview the sbatch command(s) without submitting
python run.py data run --source glass_modis --step prepare --slurm --chain --dry-run

# Submit an assembly run (per-grid resource defaults from slurm_jobs.yaml's assembly_jobs:)
python run.py assemble create --config orchestration/configs/data.yaml --grid 10km --shake quad --slurm
python run.py assemble create --config orchestration/configs/data.yaml --grid 10km --slurm --dry-run
```

## 🏗️ System Architecture

### Processing Pipeline
1. **Download**: Multi-source data acquisition with retry logic
2. **Preprocess**: Temporal aggregation (Daily → Annual), spatial harmonization
3. **Assemble**: Analysis-ready datasets with consistent alignment

### Key Features
- **Unified Interface**: Single `run.py` script for all operations
- **SLURM Integration**: `data run --slurm` submits jobs directly (`orchestration/configs/slurm_jobs.yaml` resource defaults)
- **Scalable Processing**: Dask-based parallel processing
- **Data Standards**: Chunked Zarr format for efficient I/O

## 📁 Repository Structure
```
.
├── src/                     # Core Python package
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
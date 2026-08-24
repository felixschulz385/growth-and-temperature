#!/bin/bash
#SBATCH --job-name=clean-zarr
#SBATCH --output=./log/maintenance/clean-zarr/%x-%j.out
#SBATCH --error=./log/maintenance/clean-zarr/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=1day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

rm -r /scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/grid
rm -r /scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/prepared/acag/pm25/crs/ease6933/pm25.zarr
rm -r /scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/prepared/acag/pm25/crs/legacy_4326
rm -r /scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/prepared/eog/dmsp
rm -r /scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/prepared/eog/dvnl
rm -r /scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/prepared/esacci/landcover/crs/legacy_4326
rm -r /scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/prepared/ntl_harm
rm -r /scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/snl_mining
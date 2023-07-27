# Base case to study
mpiexec -n 8 python ./code/main.py -p 1.1 -m 1 -o output -t 1000
python code/plot_biomass_flux.py -p 1.1 -m 1

# Changing the slope
mpiexec -n 8 python ./code/main.py -p 1.1 -m 0.1 -o output -t 1000
python code/plot_biomass_flux.py -p 1.1 -m 0.1

mpiexec -n 8 python ./code/main.py -p 1.1 -m 0.5 -o output -t 1000
python code/plot_biomass_flux.py -p 1.1 -m 0.5

# Changing the precipitation
mpiexec -n 8 python ./code/main.py -p 1.17 -m 1 -o output -t 1000
python code/plot_biomass_flux.py -p 1.17 -m 1

mpiexec -n 8 python ./code/main.py -p 1.19 -m 1 -o output -t 1000
python code/plot_biomass_flux.py -p 1.19 -m 1

# Get growing and decaying modes
mpiexec -n 8 python ./code/main.py -p 0.99 -m 1 -o output -t 1000
python code/plot_biomass_flux.py -p 0.99 -m 1

mpiexec -n 8 python ./code/main.py -p 1.01 -m 1 -o output -t 1000
python code/plot_biomass_flux.py -p 1.01 -m 1


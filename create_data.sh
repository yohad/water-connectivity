# Create data for patterns-flux analysis
# mpiexec -n 8 python ./code/main.py -t 1000 -p 1.1 -m 0.1
# mpiexec -n 8 python ./code/main.py -t 1000 -p 1.1 -m 1.0
# mpiexec -n 8 python ./code/main.py -t 1000 -p 1.1 -m 10

# mpiexec -n 8 python ./code/main.py -t 1000 -p 1.2 -m 0.1
# mpiexec -n 8 python ./code/main.py -t 1000 -p 1.2 -m 1.0
# mpiexec -n 8 python ./code/main.py -t 1000 -p 1.2 -m 10

# mpiexec -n 8 python ./code/main.py -t 1000 -p 1.25 -m 0.1
# mpiexec -n 8 python ./code/main.py -t 1000 -p 1.25 -m 1.0
# mpiexec -n 8 python ./code/main.py -t 1000 -p 1.25 -m 10

# mpiexec -n 8 python ./code/main.py -t 1000 -p 1.3 -m 0.1
mpiexec -n 8 python ./code/main.py -t 1000 -p 1.3 -m 1.0
mpiexec -n 8 python ./code/main.py -t 1000 -p 1.3 -m 10

# Get growing and decaying modes
mpiexec -n 8 python ./code/main.py -t 500 -p 0.99 -m 1.0
mpiexec -n 8 python ./code/main.py -t 500 -p 1.00 -m 1.0
mpiexec -n 8 python ./code/main.py -t 500 -p 1.01 -m 1.0

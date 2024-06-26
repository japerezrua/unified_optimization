#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q elektro
### -- set the job Name -- 
#BSUB -J basecase
### -- ask for number of cores (default: 1) -- 
#BSUB -n 10
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=15GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 20GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 120:00 
### -- set the email address -- 
#BSUB -u juru@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Error_%J.err 

# here follow the commands you want to execute 
python3 main_neigh_4.py

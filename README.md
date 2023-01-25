# A 2-parameter Persistence Layer for Learning

## Instructions
First clone this repo to say $MPML. Then create a conda environment by

    conda create -n mpml python=3.9 pytorch=1.12 pyg -c pytorch -c pyg

    conda activate mpml

**Additional Dependencies:**

1. Boost
2. OpenMP

Then we need to compile mpml.

    cd $MPML
    cd zigzag
    python setup.py build install

Please follow `experiments.ipynb` for instructions on how to run the code.

## Graph Experiments
You may use `run_graph_experiment.sh` to reproduce the results in the paper. Please download the precomputed landscapes from this [link](https://drive.google.com/file/d/1WWXCk3X5aKoHTlybmCnB9YLnieqpe8Mp/view?usp=share_link) and unzip the zip file to train the model faster. After unzipping it should have a directory called `graph_landscapes`.


    ./run_graph_experiment.sh PROTEINS 

Run this script to reproduce the experiment on PROTEINS dataset.




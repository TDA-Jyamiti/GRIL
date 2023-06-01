# A 2-parameter Persistence Layer for Learning

## Group Information

This project is developed by *Anonymous Authors*
<!-- [Soham Mukherjee](https://www.cs.purdue.edu/homes/mukher26/), [Cheng Xin](https://github.com/jackal092927), [Shreyas N. Samaga](https://samagashreyas.github.io) under the [CGTDA](https://www.cs.purdue.edu/homes/tamaldey/CGTDAwebsite/) research group at Purdue University lead by [Dr. Tamal K. Dey](https://www.cs.purdue.edu/homes/tamaldey/). -->


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

## Acknowledgements

This codebase heavily uses `Fast Computation of Zigzag Persistence` authored by [Tao Hou](https://taohou01.github.io). The repository for FastZigzag can be found here [https://github.com/taohou01/fzz](https://github.com/taohou01/fzz). 

## Citation
The paper is under review in ICML TAGML 2023 Workshop. More details coming soon.
![GRIL as topological discriminator!](/gril_topo_discrim_img.png "GRIL as topo discriminator")







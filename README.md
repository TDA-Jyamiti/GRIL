# A 2-parameter Persistence Layer for Learning

This codebase contains implementation of Generalized Rank Invariant Landscape (GRIL). The accompanying paper can be found here [GRIL: A -parameter Persistence Based Vectorization for Machine Learning](https://arxiv.org/pdf/2304.04970). 

## Group Information

This project is developed by [Soham Mukherjee](https://www.cs.purdue.edu/homes/mukher26/), [Cheng Xin](https://github.com/jackal092927), [Shreyas N. Samaga](https://samagashreyas.github.io) under the [CGTDA](https://www.cs.purdue.edu/homes/tamaldey/CGTDAwebsite/) research group at Purdue University lead by [Dr. Tamal K. Dey](https://www.cs.purdue.edu/homes/tamaldey/).

## Acknowledgements

This codebase heavily uses `Fast Computation of Zigzag Persistence` authored by [Tao Hou](https://taohou01.github.io). The repository for FastZigzag can be found here [https://github.com/taohou01/fzz](https://github.com/taohou01/fzz). The software is based on the following paper [Fast Computation of Zigzag Persistence](https://arxiv.org/pdf/2204.11080.pdf). 


## Instructions
First clone this repo to say $MPML. Then create a conda environment by

    conda create -n mpml python=3.9 pytorch=1.12 pyg -c pytorch -c pyg

    conda activate mpml

**Additional Dependencies:**

1. Boost
2. OpenMP

Then we need to compile mpml.

    cd $MPML
    cd gril
    python setup.py build install

Please follow `experiments.ipynb` for instructions on how to run the code. You should be able to reproduce the code.

![GRIL as topological discriminator!](/gril_topo_discrim_img.png "GRIL as topo discriminator")

## Graph Experiments
You may use `run_graph_experiment.sh` to reproduce the results in the paper. Please download the precomputed landscapes from this [link](https://drive.google.com/file/d/1WWXCk3X5aKoHTlybmCnB9YLnieqpe8Mp/view?usp=share_link) and unzip the zip file to train the model faster. After unzipping it should have a directory called `graph_landscapes`.


    ./run_graph_experiment.sh PROTEINS 

Run this script to reproduce the experiment on PROTEINS dataset.

## License

THIS SOFTWARE IS PROVIDED "AS-IS". THERE IS NO WARRANTY OF ANY KIND. NEITHER THE AUTHORS NOR PURDUE UNIVERSITY WILL BE LIABLE FOR ANY DAMAGES OF ANY KIND, EVEN IF ADVISED OF SUCH POSSIBILITY.

This software was developed (and is copyrighted by) the CGTDA research group at Purdue University. Please do not redistribute this software. This program is for academic research use only. This software uses the Boost and phat library, which are covered under their own licenses.


## Citation
The paper is under review in ICML TAGML 2023 Workshop. More details coming soon.








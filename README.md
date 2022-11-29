# sparse-anova

This repository contains Code belonging to the master thesis "Deep Learning für Optionsbewertung mit dünnen Gittern".

## Quick start 

Go to colab.google.com and click "open notebook" and paste the URL of this 
repository. Then choose "notebooks/colab-sparse.ipynb".  
After that execute the notebook.

## Running locally

I **heavily** recommend running the project in a Linux environment, either 
on Ubuntu etc. or WSL.
Clone this repository and create a Python virtual environment in the 
root path of this project.  
Make sure that a C++ compiler and cmake are installed. Open a terminal.

On Ubuntu based distros.
``bash
sudo apt install g++ cmake
``

Then in the python virtual environment:

``bash
pip install --user -r requirements.txt
``
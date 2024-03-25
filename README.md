# Macro-hop
The code was built based on Reinvent (https://github.com/MolecularAI/Reinvent) Thanks a lot for their sharing.

![](pictures/Macro-hop.png)

## Getting Started

### Installation
-------------
1. Set up conda environment and clone the github repo
2. Open a shell, and go to the repository and create the Conda environment:
```
$ conda env create -f environment.yaml
$ conda activate Macro-hop
$ cd reinvent_functions
$ pip install reinvent_function-0.0.8.1-py3-none-any.whl
```


## Analyse the results

1. tensorboard --logdir "progress.log"

    progress.log is the "logging_path" in template.json

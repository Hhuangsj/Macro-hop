# Macro-hop
The code was built based on Reinvent (https://github.com/MolecularAI/Reinvent) Thanks a lot for their sharing.

Pictures

## Getting Started

### Installation
-------------
1. Set up conda environment and clone the github repo
2. Open a shell, and go to the repository and create the Conda environment:
```
$ conda env create -f environment.yaml
$ conda activate Macro-hop
$ cd reinvent_functions
$ pip install 
```

## Usage
1. Edit template Json file (for example in result/LINK_invent/BTK/template.json).

   Templates can be manually edited before using. The only thing that needs modification for a standard run are the file and folder paths. Most running modes produce logs that can be monitored by tensorboard
2. python input.py template.json

## Analyse the results

1. tensorboard --logdir "progress.log"

    progress.log is the "logging_path" in template.json

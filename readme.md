# GTrans
This repository contains data and code for our ASE 2022 paper "Code comment generation based on graph neural network enhanced transformer model for code understanding in open-source software ecosystems".
We provide the log files while training and evaluating the models in ours-log-files directory. You can find the results and examples that we provided in our paper. 

### Data:
You can parse the original dataset to the graph format by [Parsers](https://github.com/CoderPat/structured-neural-summarization/tree/master/parsers).

You can also get the processed data from [google drive](https://drive.google.com/drive/folders/17-fksV8qPFR3JRgz0t3UXN6dfyJ6FxZX?usp=sharing).

### Training/Testing Models:

```
$ cd scripts/DATASET_NAME
```

where, choices for DATASET_NAME are ["java","python"]

To train/evealuate the GTrans model, run:

```
$ bash GTrans.sh 0 code2jdoc
```
where, 0 means GPU_ID. 

#### Running experiments on CPU/GPU/Multi-GPU
- If GPU_ID is set to -1, CPU will be used.
- If GPU_ID is set to one specific number, only one GPU will be used.
- If GPU_ID is set to multiple numbers (e.g., 0,1,2), then parallel computing will be used.

### Generated log files
While training and evaluating the models, a list of files are generated inside a `DATASET_NAME-tmp` directory. The files are as follows.
- **MODEL_NAME.mdl**
  - Model file containing the parameters of the best model.
- **MODEL_NAME.mdl.checkpoint**
  - A model checkpoint, in case if we need to restart the training.
- **MODEL_NAME.txt**
  - Log file for training.
- **MODEL_NAME.json**
  - The predictions and gold references are dumped during validation.
- **MODEL_NAME_test.txt**
  - Log file for evaluation (greedy).
- **MODEL_NAME_test.json** 
  - The predictions and gold references are dumped during evaluation (greedy).
- **MODEL_NAME_beam.txt**
  - Log file for evaluation (beam).
- **MODEL_NAME_beam.json**
  - The predictions and gold references are dumped during evaluation (beam).
  


### Acknowledgement
We borrowed and modified code from [NeuralCodeSum](https://github.com/wasiahmad/NeuralCodeSum), [ggnn.pytorch](https://github.com/chingyaoc/ggnn.pytorch). We would like to expresse our gratitdue for the authors of these repositeries.

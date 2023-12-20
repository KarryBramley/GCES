# Introduction 
ESGE is a exemplar selection method for continual learning of code intelligence models.


# Data
Our processed datasets can be downloaded in [Zenodo](https://zenodo.org/records/10409134).


# Run the code
Reproduce the results of our method and each baseline.


```markdown
|-- REPEAT
    |-- sum
    |   |-- CodeBERT
    |   |   |-- run_fineune.sh
    |   |   |-- run_emr.sh
    |   |   |-- run_ewc.sh
    |   |   |-- run_multitask.sh
    |   |   |-- run_ours.sh
    |   |   |-- ...
    |   |-- CodeT5
    |   |   |-- run_fineune.sh
    |   |   |-- run_emr.sh
    |   |   |-- run_ewc.sh
    |   |   |-- run_multitask.sh
    |   |   |-- run_ours.sh
    |   |   |-- ...
    |-- svd
    |   |-- CodeBERT
    |   |   |-- ...
    |   |-- CodeT5
    |       |-- ...
    |-- clone
    |   |-- CodeBERT
    |   |   |-- ...
    |   |-- CodeT5
    |       |-- ...
```

For example, if you want to reproduce the results of code summarization on CodeBERT, you can first move to the direcotory

```bash
cd sum/CodeBERT
```

Please first modify the data and model directory. You can also change the model's hyperparameter in each bash file. 


Normal Finetune:

```bash
bash run_finetune.sh
```


EMR method:

```bash
bash run_emr.sh
```


EWC method:

```bash
bash run_ewc.sh
```


Upper bound:

```bash
bash run_multitask.sh
```


Upper bound:

```bash
bash run_ous.sh
```






















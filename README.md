# Sila: A WIP Framework for Training, Data Labelling, and Synthetic Data Generation built to run on AWS Parallel Cluster

## This repository serves to consolidate and create  cutting edge techniques in data synthesis, filtration, and inference. Further, we hope to package everything so that anyone can leverage computer clusters and run everything through CLI.

## N.B. ternary implementation repo will be released soon - we're also fleshing out documentation + code here, stay tuned
### Documentation
- [ ] Starting up a Slurm scheduled cluster using the AWS CLI
- [ ] Running a containerized training job with SLURM
- [ ] Building a model for batched offline inference with TensorRT LLM
- [x] Running a containerized data annotation job
- [ ] Running a containerized synthetic data generation job


### 1. Data Labeling + Quality Classifiers 

#### 1.1 Generate annotations data quality classifier - distilization  
Leverages [TensorRT LLM](https://github.com/NVIDIA/TensorRT-LLM) to perform batched inference given a prompt, model, and data. The following script is based off of the `run.py` sample code located in the `/examples/` directory. The same runtime flags for the file can be used with the addition of:

* `--prepend_system_prompt`: Prepends text to the provided sample to help the model generate an output
* `--append_system_prompt`: Appends text to the provided sample to help the model generate an output
* `--output_pkl`: The path and file name of the pickle file where the tuples of prompt and output should be written to

* edit `batched_tensorRT.py` and `merge_data_subsets.py`
```bash
python batched_tensorRT.py 
```

#### 1.2 Finetune model for data quality regression
Currently predicts education value of code snippets (labels are 0-5)
* edit `train_edu_bert.py`
```bash
--base_model_name="Snowflake/snowflake-arctic-embed-m" \  # BERT-like base model
--dataset_name="https://huggingface.co/datasets/kaizen9/starcoder_annotations" \  # Llama3.1 70B -annotated eduational value dataset
--target_column="score" 
```
* run the training script on a SLURM cluster:
```bash
sbatch train_edu_bert.slurm
```

#### 1.3 Label Dataset with the educational scores with model
    
```bash
sbatch run_edu_bert.slurm
```

### 2.Synthetic Data Generation 

Coming soon!


### Appendix

Classifier code repurposed from huggingface/cosmopediav2/classifier

You can find our StarCoder Dataset Annotations ([here](https://huggingface.co/datasets/kaizen9/starcoder_annotations))

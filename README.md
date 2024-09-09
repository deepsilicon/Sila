# Sila: A WIP Framework for Training, Data Labelling, and Synthetic Data Generation built to run on AWS Parallel Cluster

### Documentation
[] Starting up a Slurm scheduled cluster using the AWS CLI
[] Running a containerized training job with SLURM
[] Building a model for batched offline inference with TensorRT LLM
[] Running a containerized data annotation job
[] Running a containerized synthetic data generation job


### 1. Finetune a model for educational value regression

* edit `train_edu_bert.slurm`
```bash
--base_model_name="Snowflake/snowflake-arctic-embed-m" \  # BERT-like base model
--dataset_name="HuggingFaceTB/LLM_juries_fineweb_430k_annotations" \  # Llama3-annotated eduational value dataset
--target_column="score" 
```
* run the training script on a SLURM cluster:
```bash
sbatch train_edu_bert.slurm
```

### 2. Annotate a dataset with the educational scores predicted by the model
    
```bash
sbatch run_edu_bert.slurm
```

### Appendix

You can find our StarCoder Dataset Annotations ([here](https://huggingface.co/datasets/kaizen9/starcoder_annotations))

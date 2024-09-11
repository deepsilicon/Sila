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

#### 1.1 Generate Annotations to Create a Data Quality Classifier - Distilliation  
Leverages [TensorRT LLM](https://github.com/NVIDIA/TensorRT-LLM) to perform batched inference given a prompt, model, and data. The following script is based off of the `run.py` sample code located in the `/examples/` directory. The same runtime flags for the file can be used with the addition of:

* `--prepend_system_prompt`: Prepends text to the provided sample to help the model generate an output
* `--append_system_prompt`: Appends text to the provided sample to help the model generate an output
* `--output_pkl`: The path and file name of the pickle file where the tuples of prompt and output should be written to

First edit `batched_tensorRT.py` and `merge_data_subsets.py` if they do not fufill your needs. Then run:
```bash
run.py \
    --engine_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
    --tokenizer_dir ./tmp/LongAlpaca-70B/
    --input_file ./samples/code_samples.pkl
    --prepend_system_prompt "Is this good code?"
    --append_system_prompt "Rate it from 1-5: "
    --output_pkl ./generated_data/analyzed_code_samples.pkl
```
To run in a multinode environment, run:
```bash
mpirun -n <number of GPUs on node> --allow-run-as-root run.py \
    --engine_dir ./tmp/llama/70B/trt_engines/fp16/8-gpu/ \
    --tokenizer_dir ./tmp/LongAlpaca-70B/
    --input_file ./samples/code_samples.pkl
    --prepend_system_prompt "Is this good code?"
    --append_system_prompt "Rate it from 1-5: "
    --output_pkl ./generated_data/analyzed_code_samples.pkl
```

Also consider using a better prompt than the ones in the examples above, or our default prompt :)

#### 1.2 Finetune Model for Data Quality Regression
Currently predicts education value of code snippets (labels are 0-5)
* edit `train_edu_bert.py`
```bash
--base_model_name="Snowflake/snowflake-arctic-embed-m" \  # BERT-like base model
--dataset_name="https://huggingface.co/datasets/kaizen9/starcoder_annotations" \  # Llama3.1 70B -annotated eduational value dataset
--target_column="score" 
```
* Run the training script on a SLURM cluster:
```bash
sbatch train_edu_bert.slurm
```

#### 1.3 Label Dataset with the Educational Scores Predicted by the Model
    
```bash
sbatch run_edu_bert.slurm
```

### 2.Synthetic Data Generation 

Coming soon!


### Appendix

Classifier code repurposed from huggingface/cosmopediav2/classifier

You can find our StarCoder Dataset Annotations ([here](https://huggingface.co/datasets/kaizen9/starcoder_annotations))

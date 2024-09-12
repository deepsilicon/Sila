import wandb

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset, ClassLabel
import numpy as np
import evaluate
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix


def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 4).astype(int)+1
    # preds = np.round(logits.squeeze()).astype(int)

    labels = np.round(labels.squeeze()).astype(int)
    precision = precision_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }


def main(args):
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name)

    dataset = load_dataset(
        args.dataset_name, split="train", cache_dir="/tmp/dataset/cache/", num_proc=8
    )
    # dataset = dataset.shuffle(seed=42).select(range(1000))

    dataset = dataset.map(
        #lambda x: {args.target_column: np.clip(int(x[args.target_column]), 0, 5)},
        lambda x: {args.target_column: int(x[args.target_column])-1},

        num_proc=8,
    )

    dataset = dataset.cast_column(
        args.target_column, ClassLabel( names=[str(i) for i in range(0, 5)])
    )
    dataset = dataset.train_test_split(
        train_size=0.9, seed=42, stratify_by_column=args.target_column
    )
    #config = AutoConfig.from_pretrained('Snowflake/snowflake-arctic-embed-m-long')
    #print(type(config))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_name,
        #config = config,
        num_labels=1,
        classifier_dropout=0.0,
        hidden_dropout_prob=0.0,
        output_hidden_states=False,
        #rotary_scaling_factor=2,
    )
    
    #config = AutoConfig.from_pretrained(
      #  "Snowflake/arctic-embed-m-long",
       # trust_remote_code=True,
      #  num_labels=1,
       #rotary_scaling_factor=2,
     #  auto_map={

    #    "AutoModelForSequenceClassification": "Snowflake/arctic-embed-m-long--modeling_hf_nomic_bert.NomicBertForSequenceClassification"
   #     }
   # )
   # model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name,
        model_max_length=min(model.config.max_position_embeddings,512),
        # model_max_length = 2048
       )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(examples):
        batch = tokenizer(examples["code"], truncation=True)
        batch["labels"] = np.float32(examples[args.target_column])
        return batch

    dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for param in model.bert.encoder.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        hub_model_id=args.output_model_name,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=50,
        learning_rate=3e-4,
        num_train_epochs=20,
        seed=0,
        per_device_train_batch_size=2048,
        per_device_eval_batch_size=2048,
        eval_on_start=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=True,
        push_to_hub=True,
        report_to="wandb",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        #callbacks=[LossPrintingCallback],
    )

    trainer.train()


    trainer.save_model(os.path.join(args.checkpoint_dir, "final"))
    eval_predictions = trainer.predict(dataset["test"])
    preds = np.round(eval_predictions.predictions.squeeze()).clip(0, 4).astype(int)+1
    labels = np.round(eval_predictions.label_ids.squeeze()).astype(int)

    # Compute metrics using the compute_metrics function
    metrics = compute_metrics((eval_predictions.predictions, eval_predictions.label_ids))

    # Print confusion matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix after training:\n" + str(cm))

    # Optionally print computed metrics
    print("Metrics after training:\n", metrics)
    wandb.finish()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", type=str, default="Snowflake/snowflake-arctic-embed-l"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="kaizen9/starcoder_annotations",
    )
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/tmp/star_edu_score/bert_snowflake_regression",
    )
    parser.add_argument(
        "--output_model_name", type=str, default="kaizen9/starcoder-scorer"
        )

    parser.add_argument(
        "--wandb_project_name", type=str, default="edu_classify"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default="starcoder"
    )
    args = parser.parse_args()

    main(args)

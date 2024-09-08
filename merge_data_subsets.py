import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import pickle
import random
import csv 
import tqdm
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

def load_and_save_mixed_data(languages, num_samples_list, file_prefix, num_batches):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    system_prompt = 'Below is a code extract. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion: - Add 1 point if the program contains valid code, even if it\'s not educational, like boilerplate code, configs, and niche concepts. - Add another point if the program addresses practical concepts, even if it lacks comments. - Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., deep learning). The code should be well-structured and contain some comments. - Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a coding course section. - Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments. The extract: '
    system_prompt_end = '\n After examining the extract: - Briefly justify your total score, up to 100 words. - Conclude with the score using the format: "Educational score: <total points>"'

    max_tokens = 3000  # tensorrt-llm max supported tokens - system prompt
    overflow_samples = 0
    all_samples = []
    if len(languages) != len(num_samples_list):
        raise ValueError("The languages and num_samples_list must be the same length")
    samples = []
    ####Loading in postive cases from High QUality datasets
    dataset_pos = datasets.load_dataset(
            path = "yuxiang630/hqcode",
            split = 'train',
            streaming = True
            )
    for i, item in enumerate(dataset_pos):
        tokens = tokenizer.encode(item['text'])
        if len(tokens) <= max_tokens:
            samples.append( item['text'])
        else:
            overflow_samples +=1
    all_samples.extend(samples)
    print("done loading HQdataset this many overflow:  ", overflow_samples)
    samples = []
    dataset_pos2 = datasets.load_dataset(
            path = "ise-uiuc/Magicoder-OSS-Instruct-75K",
            split = 'train',
            streaming = True
            )
    for i, item in enumerate(dataset_pos2):

        solution = item['solution']

        # Find the index of the first newline character
        
        first_newline = solution.find('\n')

         # Find the index of the last newline character
        last_newline = solution.rfind('\n')

        if first_newline != -1 and last_newline != -1 and first_newline < last_newline:
            # Remove both the first and last lines
            solution = solution[first_newline + 1:last_newline]

        tokens = tokenizer.encode(solution)
        if len(tokens) <=max_tokens:
            samples.append( solution)
        else:
            overflow_samples +=1
    all_samples.extend(samples)
    print("done loading OSS Dataset: this many overflow", overflow_samples )

    for language, num_samples in zip(languages, num_samples_list):
        # Load the dataset for the specified language
        dataset = datasets.load_dataset(
            path="bigcode/starcoderdata",
            data_dir=language,
            split='train',
            streaming=True
        )
        samples = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            content = item['content']
            first_newline = content.find('\n')  # Find the index of the first newline character

            if first_newline != -1:
                first_line = content[:first_newline]
                rest_of_content = content[first_newline + 1:]  # Slice the string after the first newline

                # Check if the first line contains both '<' and '>'
                if '<' in first_line and '>' in first_line:
                    content = rest_of_content
                else:
                    content = content  # Keep the original content if the condition is not met
            else:
                content = ""  # Handle the case where there's no newline
            # Check if the sample is within the token limit
            tokens = tokenizer.encode(content)
            if len(tokens) <= max_tokens:
                samples.append(content)
            else:
                tokens = tokens[:max_tokens]
                content = tokenizer.decode(tokens, skip_special_tokens=True)
                samples.append( content)
                #overflow_samples +=1
                #print(f"TRUNCATED a sample from {language} due to exceeding token limit")
        all_samples.extend(samples)

    print("done loading Starcoder, this many overflowed: ",overflow_samples )
    random.shuffle(all_samples)

    print(len(all_samples))
    # Split the samples into batches
    batch_size = len(all_samples) // num_batches
    for i in tqdm.tqdm(range(num_batches)):
        batch_samples = all_samples[i * batch_size: (i + 1) * batch_size]
        #file_path = f'{file_prefix}_batch_{i+1}.txt'

        #with open(file_path, 'w', encoding='utf-8') as file:
        #    for item in batch_samples:
        #        file.write(system_prompt + item + system_prompt_end + '\n')

        #file_path = f'{file_prefix}_batch_{i+1}.csv'
        #with open(file_path, 'w', newline='', encoding='utf-8') as file:
        #    writer = csv.writer(file)
        #    # Write data rows
        #    for item in batch_samples:
        #        # writer.writerow(tokenizer.encode(system_prompt + item+ system_prompt_end))
        #        writer.writerow([system_prompt + item +system_prompt_end])
        #        #writer.writerow([item])
                
        file_path = f'{file_prefix}_batch_{i+1}.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(batch_samples, file)

        print(f'Saved {len(batch_samples)} samples to {file_path}')

    # Handle any remaining samples if they don't fit evenly into batches
    if len(all_samples) % num_batches != 0:
        remaining_samples = all_samples[num_batches * batch_size:]
        #file_path = f'{file_prefix}_batch_{i+1}.csv'

        #with open(file_path, 'w', newline='', encoding='utf-8') as file:
        #    writer = csv.writer(file)
        #    for item in remaining_samples:
        #        writer.writerow([item])
        
        #file_path = f'{file_prefix}_batch_{num_batches+1}.pkl'
        #with open(file_path, 'wb') as file:
        #    pickle.dump(remaining_samples, file)
        print(f'Didnt Save {len(remaining_samples)} remaining samples to {file_path}')

languages = ["java", "javascript", "c", "php", "python", "cpp", "c-sharp", "typescript", "go"]
#num_samples_list = [2275336, 1648951, 599873, 1245074, 1013248, 405291, 628958, 364726, 145763]# sums up to 5 billion tokens
num_samples_list =[191268, 138613, 50426, 104662, 85175, 34069, 52871, 30659, 12253]#sums up to 700000 samples
#num_samples_list = [1, 1, 1, 1, 11, 1, 1, 1, 1]
load_and_save_mixed_data(languages, num_samples_list, '/fsx/data/mixed_sample_4000_no_prompt', 4)

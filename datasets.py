from datasets import load_dataset

# Load the dataset from Hugging Face
# Credits to : EricLu/SCP-116K
#dataset_0 = load_dataset('EricLu/SCP-116K', split = "train", trust_remote_code=True) 

# Filter the dataset
dataset_eng = dataset_0.filter(
    lambda example: example['domain'] in ['Applied Mathematics'] and
                    example['is_qwq_solution_same_with_matched_solution'] == True and
                    example['is_o1_solution_same_with_matched_solution'] == True
)

# Select and Rename the required columns
dataset_eng_filtred = dataset_eng.select_columns(['problem', 'matched_solution', 'qwq_solution'])
dataset_eng_filtred = dataset_eng_filtred.rename_columns({
    'problem': 'question',
    'matched_solution': 'response',
    'qwq_solution': 'CoT'
})

# Push the filtered dataset to the Hugging Face Hub
dataset_eng_filtred.push_to_hub(f"your_hf_name/dataset-CoT-Applied-Mathematics-{len(dataset_eng_filtred)}")

import json
import os
import random


def transform_and_split_dataset(input_json_path="input.json", output_base_dir="output_dataset", train_split=0.8, val_split=0.1):
    """
    Loads data from an input JSON file (containing a list of objects with keys
    'id', 'main_question', 'finegrained_questions', 'relevant_arxiv_ids'),
    transforms it into Hugging Face dataset format, splits it into
    train, validation, and test sets, and saves them into separate
    folders as JSON Lines (data.jsonl).

    Args:
        input_json_path (str): Path to the input JSON file (must contain a list).
        output_base_dir (str): The base directory where train/validation/test folders will be created.
        train_split (float): Proportion of data for the training set.
        val_split (float): Proportion of data for the validation set. Test split is inferred.
    """
    transformed_data = []
    num_questions_to_sample = 10
    original_data_list = []

    # --- Define the ACTUAL keys found in the input JSON ---
    key_id = 'id'
    key_main_q = 'main_question'
    key_sub_qs = 'finegrained_questions'
    key_refs = 'relevant_arxiv_ids'
    # --- Define the required keys for the check ---
    expected_keys = [key_id, key_main_q, key_sub_qs, key_refs]


    print(f"Reading data from: {input_json_path}")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as infile:
            original_data_list = json.load(infile) # Load the entire JSON structure
            if not isinstance(original_data_list, list):
                print(f"Error: Input JSON file '{input_json_path}' does not contain a list at the root.")
                return

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{input_json_path}': {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file reading: {e}")
        return

    print(f"Successfully loaded {len(original_data_list)} records from the JSON list.")

    # Process each record from the loaded list
    processed_count = 0
    skipped_count = 0
    for original_record in original_data_list:
        try:
            # --- Check for missing keys using the ACTUAL key names ---
            missing_keys = [k for k in expected_keys if k not in original_record]
            if missing_keys:
                record_identifier = original_record.get(key_id, f"Record with keys: {list(original_record.keys())}")
                # print(f"Skipping record due to missing keys: {missing_keys}. Record identifier: {record_identifier}")
                skipped_count += 1
                continue

            # --- Extract data using the ACTUAL key names ---
            record_id = original_record[key_id]
            main_question_text = original_record[key_main_q]
            all_questions = original_record[key_sub_qs]
            arxiv_ids = original_record[key_refs]

            # --- Check if the questions field is a list ---
            if not isinstance(all_questions, list):
                 print(f"Skipping record {record_id} because '{key_sub_qs}' field is not a list.")
                 skipped_count += 1
                 continue

            # --- Sample questions ---
            if len(all_questions) > num_questions_to_sample:
                sampled_questions = random.sample(all_questions, num_questions_to_sample)
            else:
                sampled_questions = all_questions # Take all if fewer than N

            # Format assistant content (join sampled questions with newlines)
            assistant_content = "\n".join(sampled_questions)

            # --- Format messages using the extracted main question text ---
            messages = [
                {
                    "role": "user",
                    "content": f"Your role is to decompose the following question into subquestions that will be searched on Google in order to find relevant Arxiv papers related to the subject: '{main_question_text}'"
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ]

            # --- Create the new record structure using standard output keys ---
            # Note: We use 'id' and 'references' as the *output* keys
            #       even though the input keys were different.
            new_record = {
                "id": record_id,
                "messages": messages,
                "references": arxiv_ids
            }
            transformed_data.append(new_record)
            processed_count += 1

        except Exception as e:
            # Catch potential errors during processing of individual records
            print(f"Error processing record {original_record.get(key_id, 'Unknown ID')}: {e}")
            skipped_count += 1


    if not transformed_data:
        print(f"No valid data processed after iterating through the list ({skipped_count} skipped). Exiting.")
        return

    print(f"Successfully processed {processed_count} records into the target format ({skipped_count} skipped).")

    # Shuffle the data before splitting
    random.shuffle(transformed_data)

    # Calculate split indices
    total_records = len(transformed_data)
    train_end = int(train_split * total_records)
    val_end = train_end + int(val_split * total_records)

    # Split the data
    train_data = transformed_data[:train_end]
    val_data = transformed_data[train_end:val_end]
    test_data = transformed_data[val_end:]

    print(f"Splitting data: Train ({len(train_data)}), Validation ({len(val_data)}), Test ({len(test_data)})")

    # Define output directories
    train_dir = os.path.join(output_base_dir, "train")
    val_dir = os.path.join(output_base_dir, "validation")
    test_dir = os.path.join(output_base_dir, "test")

    # Create directories and write data as JSON Lines (.jsonl)
    for data_split, dir_path in [(train_data, train_dir), (val_data, val_dir), (test_data, test_dir)]:
        os.makedirs(dir_path, exist_ok=True)
        output_file_path = os.path.join(dir_path, "data.jsonl")
        print(f"Writing {len(data_split)} records to: {output_file_path} (JSON Lines format)")
        try:
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                for record in data_split:
                    # Write each record as a separate JSON line
                    outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
        except IOError as e:
            print(f"Error writing to {output_file_path}: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during writing to {output_file_path}: {e}")

    print("Dataset transformation and splitting complete.")


transform_and_split_dataset(input_json_path="export/cluster_data.json", output_base_dir="dataset")
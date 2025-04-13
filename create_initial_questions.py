import json
import os

import tqdm

file_path = "data/arxiv-metadata-oai-snapshot.json"
count = 0

from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

exported_data = []

try:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm.tqdm(f, total=5000):
            try:
                # Attempt to parse the JSON object on the current line
                article = json.loads(line)

                # Process the article (e.g., print it)
                count += 1
                if count >= 5000:
                    break
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "developer", "content": "Your role is to write 3 questions that a person could be asking himself and whose questions would be answered in the provided abstract. You should remember that the person answering the question will not have access to the abstract, paper or study, meaning you shouldn't reference it in the question directly. You will write each question on a newline, each question starting with '- ' (example: '- {question1}\n- {question2}\n {question3}'). You will not write anything expect the questions."},
                        {"role": "user", "content": f"Abstract: {article['abstract']}"}
                    ]
                    )
                try:
                    questions = [q.strip() for q in completion.choices[0].message.content.split("- ") if q.strip() != ""]
                except:
                    continue

                article["questions"] = questions
                exported_data.append(article)
                with open("export/initial_questions.json", "w", encoding="utf-8") as f:
                    json.dump(exported_data, f, ensure_ascii=False, indent=4)

            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
                # Optionally print the line number or the problematic line itself
                # print(f"Problematic line content: {line.strip()}")
            except Exception as e:
                print(f"An unexpected error occurred processing a line: {e}")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred opening or reading the file: {e}")

print(f"\nFinished processing. Total articles processed: {count}")
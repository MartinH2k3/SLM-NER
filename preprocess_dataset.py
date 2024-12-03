from webbrowser import Error

import nltk
from nltk.tokenize import sent_tokenize
from sys import argv, exit
import re
import pandas as pd

nltk.download("punkt")
nltk.download('punkt_tab')

filepaths = []
categories = set()
if len(argv) < 2:
    print("Usage: python preprocess_dataset.py <filepath1> <filepath2> ...")
    print("Trying default paths")
    filepaths = ["data/CDR_DevelopmentSet.PubTator.txt", "data/CDR_TrainingSet.PubTator.txt", "data/CDR_TestSet.PubTator.txt"]
else:
    for arg in argv[1:]:
        filepaths.append(arg)


def stringify_sentence_entities():
    pass


def refactor_file(filepath: str):
    with open(filepath, "r") as file:
        data = file.read()

    articles = data.split("\n\n")
    modified_data = ""
    for article in articles:
        article_rows = article.split("\n")
        # first and second row contain the title and the article abstract
        article_content = article_rows[0].split("|")[2] + "\n" + article_rows[1].split("|")[2]

        modified_article_content = ""

        if len(article_rows) < 3:
            continue

        prev_index = 0
        # remaining rows are entities
        for row in article_rows[2:]:
            split_row = row.split("\t")
            if len(split_row) < 6:
                continue

            start_index = int(split_row[1])
            end_index = int(split_row[2])
            category = split_row[4]
            categories.add(category)
            modified_article_content += (article_content[prev_index:start_index]
                                         + f'<category="{category}">'
                                         + article_content[start_index:end_index]
                                         + f'</category>')
            prev_index = end_index
        modified_data += modified_article_content + "\n"

    print(categories)
    sentences = sent_tokenize(modified_data)
    max_length = 512
    sentences_refactored = []
    current_sentence = ""
    for sentence in sentences:
        if len(current_sentence) + len(sentence) < max_length:
            current_sentence += sentence
        else:
            sentences_refactored.append(current_sentence)
            current_sentence = sentence

    pattern = r'<category="(.*?)">(.*?)</category>'
    entities_in_sentences: list[str] = []
    cleaned_sentences: list[str] = []
    for sentence in sentences_refactored:
        sentence_entities = "["
        cleaned_sentence = sentence
        offset = 0
        for match in re.finditer(pattern, sentence):
            category = match.group(1)
            entity = match.group(2)
            start_index = match.start(2)
            end_index = match.end(2)
            sentence_entities += f"""{{\"category\": \"{category}\", \"entity\": \"{entity}\"}}, """
            cleaned_sentence = cleaned_sentence[:start_index - 13 - len(category) - offset] + entity + cleaned_sentence[
                                                                                                       end_index + 11 - offset:]
            offset += 13 + len(category) + 11
        cleaned_sentences.append(cleaned_sentence)
        sentence_entities = sentence_entities[:-2]
        sentence_entities += "]"
        entities_in_sentences.append(sentence_entities)

    table_rows = []
    for i in range(len(cleaned_sentences)):
        # Add each row to the list with 'user' and 'assistant' fields
        table_rows.append({
            'user': cleaned_sentences[i],
            'assistant': entities_in_sentences[i]  # Entities extracted in JSON format
        })

    filename = filepath.split("/")[-1].split(".")[0]
    df = pd.DataFrame(table_rows, columns=['user', 'assistant'])
    df.to_json(f"data/{filename}.json", orient='records')

for filepath in filepaths:
    print("Refactoring file:", filepath)
    refactor_file(filepath)

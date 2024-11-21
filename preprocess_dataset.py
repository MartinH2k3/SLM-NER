import nltk
from nltk.tokenize import sent_tokenize
from sys import argv, exit
import re
import pandas as pd

nltk.download("punkt")
nltk.download('punkt_tab')

filepaths = []
if len(argv) < 2:
    print("Usage: python preprocess_dataset.py <filepath1> <filepath2> ...")
    print("Trying default paths")
    filepaths = ["data/NCBIdevelopset_corpus.txt", "data/NCBItestset_corpus.txt", "data/NCBItrainset_corpus.txt"]
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
        article_content = article_rows[0].split("|")[2] + "\n" + article_rows[1].split("|")[2]
        modified_article_content = ""

        if len(article_rows) < 3:
            continue

        prev_index = 0
        for row in article_rows[2:]:
            split_row = row.split("\t")
            if len(split_row) < 5:
                continue

            start_index = int(split_row[1])
            end_index = int(split_row[2])
            category = split_row[4]
            modified_article_content += (article_content[prev_index:start_index]
                                         + f'<category="{category}">'
                                         + article_content[start_index:end_index]
                                         + f'</category>')
            prev_index = end_index
        modified_data += modified_article_content + "\n"

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
            sentence_entities += f"""\"{{\\"category\\": {category}, \\"entity\\": {entity}}}, \""""
            cleaned_sentence = cleaned_sentence[:start_index - 13 - len(category) - offset] + entity + cleaned_sentence[
                                                                                                       end_index + 11 - offset:]
            offset += 13 + len(category) + 11
        cleaned_sentences.append(cleaned_sentence)
        sentence_entities += "]"
        entities_in_sentences.append(sentence_entities)

    system_prompt = ("""Please identify all the named entities mentioned in the input sentence provided below. Use only the categories: SpecificDisease, DiseaseClass, CompositeMention, and Modifier. Remember, some terms might refer to broader disease classes, while others are specific diseases or composite mentions involving multiple diseases. You should only output the results in JSON format, following a similar structure to the example result provided.

    Example sentence and results:
    "A common human skin tumour is caused by activating mutations in beta-catenin."

    "\\"Results\\": [
        { \\"category\\": \\"DiseaseClass\\", \\"entity\\": \\"skin tumour\\" }
    ]"
    """)
    table_rows = []
    for i in range(len(cleaned_sentences)):
        # Add each row to the list with 'user' and 'assistant' fields
        table_rows.append({
            'system': system_prompt,
            'user': cleaned_sentences[i],
            'assistant': entities_in_sentences[i]  # Entities extracted in JSON format
        })

    filename = filepath.split("/")[-1].split(".")[0]
    df = pd.DataFrame(table_rows, columns=['system', 'user', 'assistant'])
    df.to_json(f"data/{filename}.json", orient='records')

for filepath in filepaths:
    print("Refactoring file:", filepath)
    refactor_file(filepath)
filenames = ["CDR_DevelopmentSet.PubTator.txt", "CDR_TrainingSet.PubTator.txt", "CDR_TestSet.PubTator.txt"]

def refactor(filepath: str):
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
        # remaining rows are entities
        for row in article_rows[2:]:
            split_row = row.split("\t")
            if len(split_row) < 6:
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

    with open(f"../data/{filepath}", "w") as file:
        file.write(modified_data)

for filename in filenames:
    print("Refactoring file:", filename)
    refactor(filename)
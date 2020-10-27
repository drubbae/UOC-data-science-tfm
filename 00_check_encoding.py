import os
import csv
import chardet

directory = r'D:\master\data science\semestre 4\M2.979 - tfm\data\influencers'

# now we will open a file for writing
data_file = open(r'D:\master\data science\semestre 4\M2.979 - tfm\data\raw_influencers_encodings.csv', mode='w', newline='', encoding='utf-8')

# create the csv writer object
csv_writer = csv.writer(data_file)
# csv file header
csv_writer.writerow(['filename', 'encoding'])

for filename in os.listdir(directory):
    if filename.endswith(".jsonl"):
        # open jsonl file
        with open(os.path.join(directory, filename), mode='rb') as f:
            # jsonl encoding
            result = chardet.detect(f.read())
            # writing data of csv file
            csv_writer.writerow([
                filename,
                result['encoding']
            ])
    else:
        continue

data_file.close()
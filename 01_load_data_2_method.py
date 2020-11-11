#https://github.com/Jokiph3r/Jsonl-to-csv/blob/master/jsonl-to-csv.py
import glob
import json
import csv
from flatten_json import flatten

debug = False
test = True

# path jsonl files
if test:
    input_path = r'D:\master\data science\semestre 4\M2.979 - tfm\data\influencers\test'
else:
    input_path = r'D:\master\data science\semestre 4\M2.979 - tfm\data\influencers'

# now we will open a file for writing
if test:
    output_path = r'D:\master\data science\semestre 4\M2.979 - tfm\data\01_data_test.csv'
else:
    output_path = r'D:\master\data science\semestre 4\M2.979 - tfm\data\01_data.csv'

data_file = open(output_path, mode='a', newline='', encoding='utf-8')

# create the csv writer object
csv_writer = csv.writer(data_file)

# write header of csv file
csv_writer.writerow([
    'filename',
    'author_followers',
    'author_full_name',
    'author_id',
    'author_image',
    'author_name',
    'author_url',
    'content',
    'date',
    'date_from_provider',
    'id',
    'id_from_provider',
    'image_url',
    'link',
    'location_latitude',
    'location_longitude',
    'place_country_code',
    'place_name',
    'place_street_address',
    'provider',
    'social_likes',
    'social_replies'
])

# reading all jsonl files
files = [f for f in glob.glob(input_path + "**/*.jsonl", recursive=True)]

for f in files:
    with open(f, mode='r') as F:
        for line in F:
            #flatten json files
            data = json.loads(line)
            data_1 = flatten(data)
            print(line, data_1)
            #creating csv files
            with open(output_path, mode='a', newline='', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                #headers should be the Key values from json files that make Coulmn header
                csv_writer.writerow([
                    F,
                    data_1['author_followers'],
                    data_1['author_full_name'],
                    data_1['author_id'],
                    data_1['author_image'],
                    data_1['author_name'],
                    data_1['author_url'],
                    data_1['content'].replace('\n', ' ').replace('\r', ' '),
                    data_1['date'],
                    data_1['date_from_provider'],
                    data_1['id'],
                    data_1['id_from_provider'],
                    data_1['image_url'],
                    data_1['link'],
                    data_1['location_latitude'],
                    data_1['location_longitude'],
                    data_1['place_country_code'],
                    data_1['place_name'],
                    data_1['place_street_address'],
                    data_1['provider'],
                    data_1['social_likes'],
                    data_1['social_replies']
                ])
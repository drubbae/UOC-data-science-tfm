import os
import json
import csv

directory = r'D:\master\data science\semestre 4\M2.979 - tfm\data\influencers'

# now we will open a file for writing
data_file = open(r'D:\master\data science\semestre 4\M2.979 - tfm\data\raw_influencers.csv', mode='w', newline='', encoding='utf-8')

# create the csv writer object
csv_writer = csv.writer(data_file)

i = 0

for filename in os.listdir(directory):
    if filename.endswith(".jsonl"):
        # open jsonl file
        with open(os.path.join(directory, filename), mode='r') as f:
            # iterate over jsonl file lines
            for line in f:
                # parse json string
                data = json.loads(line)
                print(filename, data)
                if i == 0:
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
                i += 1
                # write data of csv file
                csv_writer.writerow([
                    filename,
                    data['author']['followers'],
                    data['author']['full_name'],
                    data['author']['id'],
                    data['author']['image'],
                    data['author']['name'],
                    data['author']['url'],
                    data['content'].encode('utf-8'),
                    data['date'],
                    data['date_from_provider'],
                    data['id'],
                    data['id_from_provider'],
                    data['image_url'],
                    data['link'],
                    data['location']['latitude'],
                    data['location']['longitude'],
                    data['place']['country_code'],
                    data['place']['name'],
                    data['place']['street_address'],
                    data['provider'],
                    data['social']['likes'],
                    data['social']['replies']
                ])
    else:
        continue

data_file.close()
import csv

TXT = 'D1-new/d1.txt'
CSV = 'D1-new/d1.dat'

fields = ['user_id', 'item_id', 'ratings', 'timestamp']

with open(TXT, 'r') as file:
    Lines = file.readlines()
    file.close()

rows = []

for line in Lines:
    line = line.strip('\n').split()
    user = line[0]
    items = line[1:]
    for item in items:
        rows.append({
            'user_id' : user,
            'item_id' : item,
            'ratings' : 1,
            'timestamp' : 1000
        })

with open(CSV, 'w', newline='') as file:
    csvwritter = csv.DictWriter(file, fieldnames= fields)
    csvwritter.writerows(rows)
    file.close()
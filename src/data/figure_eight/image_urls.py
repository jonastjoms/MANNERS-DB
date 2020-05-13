#!/usr/bin/python
import os, sys
import csv

# Open a file
path = "/Users/jonastjomsland/Dissertation/data/screenshots"
dirs = os.listdir(path)

dict = {}

with open('data/features4.csv','r') as csvinput:
    with open('images4.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)
        all = []
        row = next(reader)
        row[43] = "image_url"
        all.append(row)
        i = 0
        for row in reader:
            row[43] = "https://acsdissertation.s3-eu-west-1.amazonaws.com/" + row[1][17:]
            all.append(row)
            i += 1
        writer.writerows(all)

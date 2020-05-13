#!/usr/bin/python
# Load csv file and write to new csv file:
import os, sys
import csv

# Open cleaned file from labeling:
with open('cleaned_step1.csv','r') as csvinput:
    # Open file to write to
    with open('data', 'w') as csvoutput:
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

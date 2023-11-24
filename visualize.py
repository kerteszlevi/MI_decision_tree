import csv

with open('train.csv', 'r') as file:
    # Create a CSV reader
    reader = csv.reader(file)

    # Iterate over each row in the CSV
    for row in reader:
        # Each row is a list of strings
        # representing the values in each column
        print(row)
        
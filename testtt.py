import csv

for i in range(3):



    with open('test.csv', 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["new", "epoch", "over", "here", "."])
            writer.writerow([1,2,3,4,5])
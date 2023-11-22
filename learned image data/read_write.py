import csv

def read_inc_images():
    filename = []
    with open('learned image data/inc_images.csv', 'r') as file:
        reader = csv.reader(file)

        for rows in reader:
            filename.extend(rows)

    return filename        


def write_inc_images(filename):
    with open('learned image data/inc_images.csv', 'w') as file:
        writer = csv.writer(file)

        writer.writerows([filename])
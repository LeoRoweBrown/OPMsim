import csv

from numpy import genfromtxt    
data = genfromtxt('refractive_index_data/SiO.txt', delimiter='\t')
print(data)


with open('refractive_index_data/SiO.txt', newline='') as csvfile:

    n_data = csv.reader(csvfile, delimiter='\t', quotechar='|')

    for row in n_data:
        print(row)
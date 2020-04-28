import os
from tqdm import tqdm

for file in ['valid', 'test']:

    data = open(f'rottentomatoes/{file}.txt', 'r').read().replace('_', '&&&').split('\n')[:-1]
    new_data = data[:]

    for row in tqdm(data):
        row = row.split()
        new_row = [str(1 + int(row[0]) % 2), row[1], row[2], str(0)]
        new_data.append(' '.join(new_row))

    open(f'rottentomatoes_clean/{file}.txt', 'w').write('\n'.join(new_data))
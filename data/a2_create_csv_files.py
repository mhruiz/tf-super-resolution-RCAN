import pandas as pd
import os

# Get all directories
directories = list(filter(os.path.isdir, os.listdir()))

# Base path for all datasets (this directory)
base_path = 'data'

# Create a new directory for csv files
if not os.path.exists('0_csvs'):
    os.mkdir('0_csvs')

# For each directory in current directory
for d in directories:
    
    if d == '0_csvs':
        continue

    # Check if current dataset was not seen before
    if not os.path.exists('0_csvs/' + d + '.csv'):
        images = os.listdir(d)
        
        # Create path list with all image paths of current dataset
        paths = [base_path + '/' + d + '/' + x for x in images]
               
        df = pd.DataFrame({'path' : paths})
        
        df.to_csv('0_csvs/' + d + '.csv', index=False)
        
        print('Saved',d + '.csv')
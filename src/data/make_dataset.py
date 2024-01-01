# make_dataset.py
import pathlib
import yaml
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def split_data(df, test_split, seed):
    # Split the dataset into train and test sets
    train, test = train_test_split(df, test_size=test_split, random_state=seed)
    return train, test

def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)

def main():

    curr_dir = pathlib.Path(__file__) #by this we are getting this file's directory
     
    home_dir = curr_dir.parent.parent.parent #with is we get the project's directory
    
    params_file = home_dir.as_posix() + '/params.yaml'
    #.as_posix() change pathlib format of path to string that python manipulate
    params = yaml.safe_load(open(params_file))["make_dataset"]
    #now read params.yaml file and take your choice data

    input_file = sys.argv[1] #we dont want to hardcode things so we take argument
    #of data file name, but this we can hardcode as having only one file
    #data_path = home_dir.as_posix() + input_file #making path to data that is in raw
    data_path = home_dir.as_posix() + '/data/raw/creditcard.csv'
    
    output_path = home_dir.as_posix() + '/data/processed'
    
    data = load_data(data_path)
    train_data, test_data = split_data(data, params['test_split'], params['seed'])
    save_data(train_data, test_data, output_path)

if __name__ == "__main__":
    main()
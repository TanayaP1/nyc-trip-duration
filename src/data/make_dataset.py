# make_dataset.py
import sys
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_raw_data(input_path: Path) ->  pd.DataFrame:
    raw_data = pd.read_csv(input_path)
    rows, columns = raw_data.shape
    return raw_data

def train_val_split(data: pd.DataFrame,
                    test_size: float,
                    random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    train_data, val_data = train_test_split(data,
                                            test_size= test_size,
                                            random_state= random_state)
    return train_data, val_data

def save_data(data: pd.DataFrame,output_path: Path):
    data.to_csv(output_path,index=False)

def main():
    # read the input file name from command
    print(len(sys.argv))
    input_file_name = sys.argv[1]
    # current file path
    current_path = Path(__file__)
    # root directory path
    root_path = current_path.parent.parent.parent

    params_file = root_path.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["make_dataset"]

    # interim data directory path
    interim_data_path = root_path / 'data' / 'interim'
    # make directory for the interim path
    interim_data_path.mkdir(exist_ok= True)
    # raw train file path
    raw_df_path = root_path / 'data' / 'raw' / 'extracted' / input_file_name
    # load the training file
    raw_df = load_raw_data(input_path= raw_df_path)
    # split the file to train and validation data
    train_df, val_df = train_val_split(data= raw_df,
                                       test_size=params['test_size'], 
                                       random_state=params['seed'])
    # save the train data to the output path
    save_data(data= train_df, output_path= interim_data_path / 'train.csv')
    # save the val data to the output path
    save_data(data= val_df, output_path= interim_data_path / 'val.csv')

if __name__ == '__main__':
    main()

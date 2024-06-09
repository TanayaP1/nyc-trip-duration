from zipfile import ZipFile
from pathlib import Path

def extract_zip_file(input_path: Path,output_path: Path):
    with ZipFile(file= input_path) as f:
        f.extractall(path= output_path)
        input_file_name = input_path.stem + input_path.suffix

def main():
    # current file path 
    current_path = Path(__file__).resolve()
    # root directory path
    root_path = current_path.parent.parent.parent
    # raw data directory path
    raw_data_path = root_path / 'data' / 'raw'
    # output path for the zip files
    output_path = raw_data_path / 'extracted'
    # make the directory for the path
    output_path.mkdir(parents=True,exist_ok=True)
    # input path for zip files
    input_path = raw_data_path / 'zipped'
    
    # extract the train and test files
    # for the train file
    extract_zip_file(input_path= input_path / 'train.zip',
                     output_path= output_path )
    # for the test file
    extract_zip_file(input_path= input_path / 'test.zip',
                     output_path= output_path)
    
if __name__ == "__main__":
    main()
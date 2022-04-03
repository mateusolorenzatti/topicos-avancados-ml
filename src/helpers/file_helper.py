import os

def get_data_file(filename):
    file_path = os.path.realpath(__file__)
    
    print(file_path)

    return
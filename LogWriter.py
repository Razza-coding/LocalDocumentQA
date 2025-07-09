import os, sys
from os import path
import re
import tqdm
import json
from datetime import datetime

class LogWriter():
    def __init__(self, log_name: str, log_folder_name: str | None = None, root_folder: str = ".", encoding: str ='utf-8'):
        # check special symbols in log name / log folder name
        if not self.__check_file_name_valid(log_name):
            self.__raise_string_error(log_name)
        if log_folder_name and not self.__check_file_name_valid(log_folder_name):
            self.__raise_string_error(log_folder_name)

        #self.log_create_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_create_time = datetime.now().strftime("%Y%m%d")
        self.log_file = f"{self.log_create_time}_{log_name}.txt"
        self.log_root = path.abspath(root_folder)
        self.log_folder = self.log_root if not log_folder_name else path.join(self.log_root, log_folder_name)
        self.__encoding = encoding
        # seperate lines
        line_length = 100
        self.__s_t_line = "-" * line_length + "\n" # thin
        self.__s_n_line = "=" * line_length + "\n" # normal
        self.__s_b_line = "#" * line_length + "\n" # bold
        self.__s_line_set = [self.__s_t_line, self.__s_n_line, self.__s_b_line]

        if not path.isdir(self.log_folder):
            os.makedirs(self.log_folder, exist_ok=True) # super make leaf folders

        if not path.isdir(self.log_folder):
            self.__raise_string_error(self.log_folder)

        pass

    def __check_file_name_valid(self, file_name: str):
        if re.search(r'[^\w\s.-]', file_name):
            return False # not valid for creating file or folder
        return True

    def remove_special_symbol(self, input: str, replacement: str ='_'):
        return re.sub(r'[^\w\s.-]', replacement, input)
    
    def __raise_string_error(error_str: str):
        raise ValueError(f"String not Valid : {error_str}")
    
    def write_log(self, log_message: str, message_section: str | None = None, message_end="\n"):
        lf_path = path.join(self.log_folder, self.log_file)
        mode = "a" if path.exists(lf_path) else "w"
        with open(file=lf_path, mode=mode, encoding=self.__encoding) as lf:
            if message_section:
                lf.write(f"\n[ {message_section} ]\n")
                lf.write(self.__s_t_line)
            lf.write(str(log_message) + message_end)
    
    def write_line(self, line_size: int = 0):
        lf_path = path.join(self.log_folder, self.log_file)
        mode = "a" if path.exists(lf_path) else "w"
        line_size = min(line_size, len(self.__s_line_set))
        with open(file=lf_path, mode=mode, encoding=self.__encoding) as lf:
            lf.write(self.__s_line_set[line_size])
        
    def __str__(self):
        return str( path.join(self.log_folder, self.log_file))
    
if __name__ == "__main__":
    writer = LogWriter("test", "test_log")
    writer.write_line(2)
    writer.write_log("create log file success", "CREATE")
    writer.write_log("write log message success", "WRITE")
    writer.write_log("write log message withou section success")
    writer.write_line(2)

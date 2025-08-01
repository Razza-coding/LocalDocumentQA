import os, sys
from os import path
import re
import tqdm
import json
from datetime import datetime

def remove_special_symbol(input: str, replacement: str ='_'):
    ''' Remove symbol not valid for creating file or folder '''
    return re.sub(r'[^\w\s.-]', replacement, input)

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
        ''' Check if string valid for creating file or folder '''
        if re.search(r'[^\w\s.-]', file_name):
            return False 
        return True  
    
    def __raise_string_error(error_str: str):
        ''' Raise error and show string '''
        raise ValueError(f"String not Valid : {error_str}")
    
    def _get_time(self) -> str:
        ''' Get current time for logging or printing '''
        return str(datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f"))
    
    def clear(self):
        ''' clear all log file content '''
        lf_path = path.join(self.log_folder, self.log_file)
        with open(file=lf_path, mode="w", encoding=self.__encoding) as lf:
            lf.write("")
    
    def write_log(self, log_message: str, message_section: str | None = None, add_time:bool = True, message_end="\n"):
        ''' 
        Write a message into log file
        If message section is given, write down Section Title with a seperate line before writing log message
        '''
        lf_path = path.join(self.log_folder, self.log_file)
        mode = "a" if path.exists(lf_path) else "w"
        with open(file=lf_path, mode=mode, encoding=self.__encoding) as lf:
            if message_section:
                if add_time:
                    time_stamp = f"[ {self._get_time()} ]"
                else:
                    time_stamp = ""
                msg_sec_title = f"\n[ {message_section} ] {time_stamp}\n"
                lf.write(msg_sec_title)
                lf.write(self.__s_t_line)
            lf.write(str(log_message) + message_end)
    
    def write_s_line(self, line_size: int = 0):
        ''' Draw a seperate line, line size switches what char is used (-, =, #) '''
        lf_path = path.join(self.log_folder, self.log_file)
        mode = "a" if path.exists(lf_path) else "w"
        line_size = min(line_size, len(self.__s_line_set))
        with open(file=lf_path, mode=mode, encoding=self.__encoding) as lf:
            lf.write(self.__s_line_set[line_size])
    
    def write_time(self):
        ''' Write current time in log file '''
        self.write_log(f"[ time ] {self._get_time()}")
        
    def __str__(self):
        ''' return log file path '''
        return str( path.join(self.log_folder, self.log_file))
    
if __name__ == "__main__":
    writer = LogWriter("test", "test_log")
    writer.write_s_line(2)
    writer.write_log("create log file success", "CREATE")
    writer.write_log("write log message success", "WRITE")
    writer.write_log("write log message withou section success")
    writer.write_time()
    writer.write_s_line(2)

import os, sys, re
import math

def CLI_print(speaker_name:str="System", message:str="", speaker_sidenote:str=""):
    """ A formatted print for beautify UI """
    name_width = max(int(math.ceil(len(speaker_name) / 5) * 5), 15)
    name_label = f"[ {speaker_name} ]"
    pm = f"{name_label:<{name_width}} | {speaker_sidenote}\n{message}"
    print(pm)
    return pm

def CLI_input():
    """ Ask User to input something """
    return input(f"{'[ User Input ]':<15} | ")

def CLI_next():
    """ Print a new line """
    print("")

if __name__ == "__main__":
    CLI_print("Test Process", "Hello World", "Test Message")
    CLI_input()
    CLI_print("Test Process", "All Passed", "Ending")
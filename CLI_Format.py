import os, sys, re
import rich
import math

def CLI_print(speaker_name:str="System", message:str="", speaker_sidenote:str=""):
    """ A formatted print for beautify UI """
    name_width = max(int(math.ceil(len(speaker_name) / 5) * 5), 15)
    name_label = f"[ {speaker_name} ]"
    pm = f"{name_label:<{name_width}} | {speaker_sidenote}\n{message}"
    rich.print(pm)
    return pm

def CLI_input():
    """ Ask User to input something """
    return input(f"{'[ User Input ]':<15} | ")

def CLI_next():
    """ Print a new line """
    rich.print("")

if __name__ == "__main__":
    python_text = """
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

p1 = Person("John", 36)

for i in range(10):
    print (i)
"""
    CLI_print("Test Process", "Hello World", "Test Message")
    CLI_print("Test Process", python_text, "Pretty Print Python")
    user_input = CLI_input()
    CLI_print("Test Process", user_input, "Show Input")
    CLI_print("Test Process", "All Passed", "Ending")
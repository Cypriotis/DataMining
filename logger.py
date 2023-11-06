import os

log_file = '/home/tofi-machine/Documents/DataMining/DataMining/log.txt'

def log(message):
    with open(log_file, 'a') as file:
        file.write(message + '\n')

def clear_log():
    if os.path.exists(log_file):
        os.remove(log_file)
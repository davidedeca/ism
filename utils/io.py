from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
                      
# usage:

# with suppress_stdout():
#    some_command

# nothing will be printed to screen


def go_up(path, n=1):
    for nn in range(n):
        path = os.path.dirname(path)
    return path


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * (float(iteration) / float(total)))
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r'+prefix+bar+percent+suffix)
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total: 
        print('\n')

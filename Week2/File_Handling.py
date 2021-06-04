import os

#This function calculates the number of white spaces and returns the value
def white_spaces(file1):
    num_white_spaces = 0

    with open(file1, 'r') as f1:
        for line in f1:
            line = line.strip(os.linesep)

            for s in line:
                if s in (os.linesep,' '):
                    num_white_spaces += 1

    return num_white_spaces


#This function creates the output text file and adds text to it
def create_output(file1, file2):
    with open(file1, 'r') as f1:
        with open(file2, 'w') as f2:
            for line in f1:

                line = line.strip(os.linesep)
                line = line.title()
                
                f2.write(line)
                f2.write("\n")

    with open(file2, 'a') as f2:
        f2.write("\nTotal number of white spaces in text file are ")
        f2.write(str(white_spaces(file1)))


def main():
    input_file = 'input.txt'
    output_file = 'output.txt'

    try:
        create_output(input_file, output_file)
    except:
        print("File not found")

main()
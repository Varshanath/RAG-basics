# Open the file in read mode ('r')
with open('c:/Users/VARSHA NATH/Projects/RAG basics/testfilereader.txt', 'r', encoding='utf-8') as file:
    # Loop through each line in the file
    for line in file:
        # print() adds a newline by default; .strip() removes extra spacing/newlines from the file
        print(line.strip())
def read_first_500_words(filename):
    # Open the file in read mode
    with open(filename, 'r', encoding='utf-8') as file:
        # Read the entire content of the file
        text = file.read()
        
        # Split the text into a list of words based on whitespace
        words = text.split()
        
        # Slice the list to get only the first 500 words
        limited_words = words[:500]
        
        # Join them back into a single string (or process them as needed)
        limited_text = ' ' .join(limited_words)
        
        return limited_text, len(limited_words)

# Usage example:
file_path = 'sample.txt'
content, count = read_first_500_words(r"c:/Users/VARSHA NATH/Projects/RAG basics/testfilereader.txt")
print(f"Successfully read {count} words:\n")
print(content)
#import chromadb
from networkx import difference
from sentence_transformers import SentenceTransformer

documents = [
    "The company reimburses employee travel expenses only when the trip is approved in advance and supported by receipts.",
    "Employees may take up to 20 days of annual leave each year, subject to manager approval and business needs.",
    "All IT systems must use strong passwords and multi-factor authentication before accessing company data from outside the office.",
    "Procurement requires vendor onboarding to be completed before any new supplier can be paid or contracted.",
    "The organization prohibits sharing confidential customer data with unauthorized personnel or external parties.",
    "All IT systems must use strong passwords and multi-factor authentication before accessing company data from outside the office."]

print ("documents:", documents[2])

#client = chromadb.PersistentClient(path="./my_chroma_db")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents).tolist()

print("embeddings:", embeddings[2])
print("embeddings length:", len(embeddings[2]))

while True:
    query1 = input("Enter the sentence number you want to compare: ")
    query2 = input("Enter the second sentence number you want to compare with: ")
    
    for i in range(len(embeddings[2])): 
        difference = embeddings[int(query1)][i] - embeddings[int(query2)][i]
        print(f"Difference in dimension {i}: {difference}")
        sum_difference = sum(abs(embeddings[int(query1)][i] - embeddings[int(query2)][i]) for i in range(len(embeddings[2])))
        print(f"Total difference between sentence {query1} and sentence {query2}: {sum_difference}")
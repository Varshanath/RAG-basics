import chromadb
import time

from networkx import difference
from sentence_transformers import SentenceTransformer

documents = [
    "The company reimburses employee travel expenses only when the trip is approved in advance and supported by receipts.",
    "Employees may take up to 20 days of annual leave each year, subject to manager approval and business needs.",
    "All IT systems must use strong passwords and multi-factor authentication before accessing company data from outside the office.",
    "Procurement requires vendor onboarding to be completed before any new supplier can be paid or contracted.",
    "The organization prohibits sharing confidential customer data with unauthorized personnel or external parties.",
    "All IT systems must use strong passwords and multi-factor authentication before accessing company data from outside the office.",
    "The difference between abc and xyz poilcies is that abc allows for flexible work hours, while xyz requires employees to adhere to a strict 9-to-5 schedule.",
    "কোম্পানিটি কর্মচারীদের ভ্রমণের খরচ শুধুমাত্র তখনই প্রতিদান (reimburse) করে, যখন সেই ট্রিপ বা ভ্রমণটি অগ্রিম অনুমোদিত হয় এবং তার স্বপক্ষে রসিদ (receipts) জমা দেওয়া হয়।",
    "Remote employees must connect to company resources only through the approved VPN client.",
    "Performance reviews are conducted twice a year and directly influence annual salary adjustments.",
    "All expense reports exceeding $500 require additional approval from the finance director.",
    "New hires must complete mandatory compliance training within their first 30 days of employment.",
    "Company laptops must be encrypted and enrolled in the mobile device management system before use.",
    "Employees are entitled to five paid sick days per year without requiring a medical certificate.",
    "Any software installed on company devices must first be approved by the IT security team.",
    "Client contracts above $100,000 require sign-off from both legal and executive leadership.",
    "Employees must report any suspected data breach to the security team within 24 hours.",
    "Parental leave provides up to 12 weeks of paid time off for eligible employees.",
    "Company credit cards may only be used for business-related expenses and require monthly reconciliation.",
    "Remote work requests must be submitted and approved by a direct manager at least one week in advance.",
    "All vendor contracts must be reviewed by the procurement team before signature.",
    "Employees are prohibited from using personal email accounts to transmit confidential company data.",
    "Overtime pay is only authorized when pre-approved by a department manager.",
    "The company provides a wellness stipend of up to $50 per month for gym memberships.",
    "Exit interviews are mandatory for all employees leaving the organization voluntarily.",
    "Access to the production database is restricted to senior engineers with two-factor authentication enabled.",
    "Business travel bookings must be made through the approved corporate travel platform.",
    "Employees must renew their security awareness certification on an annual basis.",
    "Conflicts of interest must be disclosed to HR within five business days of becoming aware of them.",
    "All company data must be backed up daily and retained for a minimum of seven years."]

print ("documents:", documents[2])


#just creating embeddings for the documents using the sentence transformer model

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents).tolist()

#creating a collection in ChromaDB to store the embeddings  
client = chromadb.EphemeralClient()
collection = client.get_or_create_collection(
    name="custom_model_collection"
)
collection.add(
    ids=[str(i) for i in range(len(documents))],
    embeddings=embeddings,
    documents=documents)

input_text = input("Enter the sentence you want to compare: ")
query_embedding = model.encode([input_text]).tolist()
query_results = collection.query(
    query_embeddings=query_embedding,n_results=3,include=["distances","documents"]
)

print("query_results:", query_results)

for doc_id, distance, document in zip(query_results["ids"][0], query_results["distances"][0], query_results["documents"][0]):
    print(f"[{doc_id}] (distance: {distance:.4f}) {document}")

"""print("embeddings:", embeddings[2])
print("embeddings length:", len(embeddings[2]))
stopwatch = time.time()
for num1 in range(0, documents.__len__()-1):
    for num2 in range(num1+1, documents.__len__()):
        query1 = num1
        query2 = num2

        #query1 = input("Enter the sentence number you want to compare: ")
        #query2 = input("Enter the second sentence number you want to compare with: ")

        for i in range(len(embeddings[2])):
            difference = embeddings[int(query1)][i] - embeddings[int(query2)][i]
            print(f"Difference in dimension {i}: {difference}")
            sum_difference = sum(abs(embeddings[int(query1)][i] - embeddings[int(query2)][i]) for i in range(len(embeddings[2])))
            print(f"Total difference between sentence {query1} and sentence {query2}: {sum_difference}")
stopwatch = time.time() - stopwatch
print(f"Time taken: {stopwatch}")"""
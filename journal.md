i have started building the RAG, also learning with it

made the 5 basic model now adding more RAG thimgs iit.

running tests 
Your branch is up to date with 'origin/main'.

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/947b7d83-e345-4442-bff2-0db9ac78e337" />


now gonna make the text loader, Its job is simple: read a .txt file and return its contents as text. Later, we'll add PDF and DOCX loaders that produce the same Document structure.

WOW

$ python -c "from src.ingestion.text_loader import load_text_file; doc = load_text_file('data/sample.txt'); print('ID:', doc.id); print('Source:', doc.source); print('Metadata:', doc.metadata); print('Content length:', len(doc.content))"
ID: sample
Source: sample.txt
Metadata: {'file_type': '.txt'}
Content length: 334
(.venv) 
Smart Te

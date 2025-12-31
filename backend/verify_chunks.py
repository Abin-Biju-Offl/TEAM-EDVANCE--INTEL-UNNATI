"""Verify enhanced chunk metadata"""
import pickle

chunks = pickle.load(open('vector_store/faiss_index.chunks.pkl', 'rb'))

print(f'Total chunks: {len(chunks)}')
print('\n=== Sample Enhanced Chunk ===')

sample = chunks[10]
print(f'Primary Page: {sample.get("primary_page")}')
print(f'Content Type: {sample.get("content_type")}')
print(f'Keywords: {sample.get("keywords")}')
print(f'Has Equations: {sample.get("has_equations")}')
print(f'Has Questions: {sample.get("has_questions")}')
print(f'Subject: {sample.get("subject")}')
print(f'Chapter: {sample.get("chapter")}')
print(f'Text preview: {sample.get("text", "")[:150]}...')

print('\n=== Content Type Distribution ===')
content_types = {}
for chunk in chunks:
    ctype = chunk.get('content_type', 'unknown')
    content_types[ctype] = content_types.get(ctype, 0) + 1

for ctype, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
    print(f'{ctype}: {count} chunks ({count/len(chunks)*100:.1f}%)')

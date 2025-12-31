import pickle

# Load chunks
with open('vector_store/faiss_index.chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

print(f"Total chunks: {len(chunks)}")
print(f"\nFirst chunk keys: {list(chunks[0].keys())}")

# Count by filename
filenames = {}
for chunk in chunks:
    fn = chunk.get('filename', 'unknown')
    filenames[fn] = filenames.get(fn, 0) + 1

print(f"\nChunks by filename:")
for fn, count in sorted(filenames.items()):
    print(f"  {fn}: {count} chunks")

# Check for English files
english_files = [fn for fn in filenames.keys() if fn.startswith('jeff')]
print(f"\nEnglish files found: {len(english_files)}")
if english_files:
    print("English files:")
    for fn in english_files:
        print(f"  - {fn}: {filenames[fn]} chunks")
else:
    print("⚠️  NO ENGLISH FILES FOUND!")

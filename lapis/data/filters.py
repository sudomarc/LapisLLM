import re


def filter_empty_docs(documents, min_length=1):
    """Filter out empty or whitespace-only documents."""
    return [doc for doc in documents if doc.strip() and len(doc.strip()) >= min_length]


def filter_by_length(documents, min_length=1, max_length=10000):
    """Filter documents by length."""
    return [doc for doc in documents if min_length <= len(doc) <= max_length]


def filter_by_language(documents, language="en", min_ratio=0.8):
    """Filter documents by language (basic heuristic)."""
    # Simple heuristic: check if most characters are in the target language
    filtered = []
    for doc in documents:
        if not doc.strip():
            continue
        # Count ASCII letters vs others
        alpha_count = sum(1 for c in doc if c.isalpha())
        if len(doc) == 0:
            continue
        ratio = alpha_count / len(doc) if len(doc) > 0 else 0
        if ratio >= min_ratio:
            filtered.append(doc)
    return filtered


def filter_duplicates(documents):
    """Remove duplicate documents (exact match)."""
    seen = set()
    unique = []
    for doc in documents:
        if doc not in seen:
            seen.add(doc)
            unique.append(doc)
    return unique


def filter_excessive_repetition(documents, max_repeats=3, min_seq=3):
    """Filter documents with excessive repetition."""
    filtered = []
    for doc in documents:
        words = doc.split()
        if len(words) < min_seq:
            filtered.append(doc)
            continue
        
        # Check for repeated sequences
        has_repetition = False
        for i in range(len(words) - min_seq + 1):
            seq = " ".join(words[i:i+min_seq])
            count = sum(1 for j in range(len(words) - min_seq + 1) if " ".join(words[j:j+min_seq]) == seq)
            if count > max_repeats:
                has_repetition = True
                break
        
        if not has_repetition:
            filtered.append(doc)
    
    return filtered


def apply_filters(documents, min_length=1, max_length=10000, 
                  language="en", filter_dups=True, filter_repetition=True):
    """Apply full filter pipeline to documents."""
    # Step 1: Remove empty
    documents = filter_empty_docs(documents, min_length=min_length)
    
    # Step 2: Filter by length
    documents = filter_by_length(documents, min_length=min_length, max_length=max_length)
    
    # Step 3: Filter by language
    documents = filter_by_language(documents, language=language)
    
    # Step 4: Remove duplicates
    if filter_dups:
        documents = filter_duplicates(documents)
    
    # Step 5: Filter excessive repetition
    if filter_repetition:
        documents = filter_excessive_repetition(documents)
    
    return documents
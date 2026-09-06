import torch


def pack_tokens(tokenized_documents, max_seq_length):
    """Pack multiple tokenized documents into fixed-length sequences.
    
    Returns a list of packed token ID tensors, each of length max_seq_length.
    """
    packed = []
    
    for doc_tokens in tokenized_documents:
        # Truncate or pad
        if len(doc_tokens) > max_seq_length:
            # Truncate
            packed.append(doc_tokens[:max_seq_length])
        else:
            # Pad with pad token (0)
            padded = doc_tokens + [0] * (max_seq_length - len(doc_tokens))
            packed.append(padded)
    
    return packed


def pack_into_batches(tokenized_data, batch_size, max_seq_length):
    """Pack tokenized data into batches."""
    packed_sequences = pack_tokens(tokenized_data, max_seq_length)
    
    # Group into batches
    batches = []
    for i in range(0, len(packed_sequences), batch_size):
        batch = packed_sequences[i:i+batch_size]
        # Pad batch to same size if needed
        if batch:
            batches.append(batch)
    
    return batches


def create_attention_mask(seq_len, device="cpu"):
    """Create a causal attention mask."""
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1
    )
    return mask
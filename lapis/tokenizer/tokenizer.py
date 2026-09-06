import json
import os


class Tokenizer:
    """LAPIS Tokenizer - supports encoding/decoding with special tokens."""
    
    def __init__(self, vocab=None, bos_token="<s>", eos_token="</s>", 
                 pad_token="<p>", unk_token="<u>"):
        """Initialize tokenizer with vocabulary.
        
        Args:
            vocab: Dict mapping token -> id. If None, uses default vocab.
            bos_token: Beginning of sentence token
            eos_token: End of sentence token
            pad_token: Padding token
            unk_token: Unknown token
        """
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self._build_default_vocab()
        else:
            self.vocab = vocab
        
        # Reverse mapping: id -> token
        self.itos = {idx: token for token, idx in self.vocab.items()}
        
        # Special token IDs
        self.bos_id = self.vocab.get(bos_token, 0)
        self.eos_id = self.vocab.get(eos_token, 1)
        self.pad_id = self.vocab.get(pad_token, 2)
        self.unk_id = self.vocab.get(unk_token, 3)
        
        # Vocab size
        self.vocab_size = len(self.vocab)
    
    def _build_default_vocab(self):
        """Build a default vocabulary."""
        vocab = {}
        
        # Add special tokens
        for i, token in enumerate([self.bos_token, self.eos_token, self.pad_token, self.unk_token]):
            vocab[token] = i
        
        # Add some base tokens (letters, digits, common)
        base_tokens = []
        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'\"()-_/@#$%^&*+=<>[]{}":
            if c not in [t for t in vocab]:
                vocab[c] = len(vocab)
        
        # Add some common words
        common_words = ["the", "and", "of", "to", "a", "in", "is", "it", "you", "that"]
        for word in common_words:
            if word not in vocab:
                vocab[word] = len(vocab)
        
        return vocab
    
    def encode(self, text, add_special_tokens=True):
        """Encode text to token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: If True, adds BOS and EOS tokens
            
        Returns:
            List of token IDs
        """
        # Simple character-level or word-level encoding
        # For now, use a simple approach: split by known tokens
        tokens = []
        
        # Check for BOS/EOS
        if add_special_tokens:
            tokens.append(self.bos_id)
        
        # Simple word-piece like encoding
        # Split by spaces and handle known words
        words = text.lower().split()
        for word in words:
            # Try to find the word in vocab
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Fall back to character encoding
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                # Add unknown token
                tokens.append(self.unk_id)
        
        if add_special_tokens:
            tokens.append(self.eos_id)
        
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: If True, removes BOS/EOS/PAD/UNK
            
        Returns:
            Decoded text string
        """
        tokens = []
        for idx in token_ids:
            if skip_special_tokens and idx in (self.bos_id, self.eos_id, self.pad_id, self.unk_id):
                continue
            if idx in self.itos:
                tokens.append(self.itos[idx])
            else:
                tokens.append(f"<unk_{idx}>")
        
        # Join tokens back to text
        text = " ".join(tokens)
        return text
    
    def batch_encode(self, texts, add_special_tokens=True):
        """Batch encode multiple texts.
        
        Args:
            texts: List of input text strings
            add_special_tokens: If True, adds BOS and EOS tokens
            
        Returns:
            List of lists of token IDs
        """
        return [self.encode(text, add_special_tokens=add_special_tokens) 
                for text in texts]
    
    def save(self, path):
        """Save tokenizer to file.
        
        Args:
            path: Directory path to save tokenizer files
        """
        os.makedirs(path, exist_ok=True)
        
        # Save vocab
        vocab_path = os.path.join(path, "tokenizer.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab": self.vocab,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "vocab_size": self.vocab_size,
            }, f, ensure_ascii=False, indent=2)
        
        # Save metadata
        metadata = {
            "tokenizer_version": "lapis-tokenizer-v1",
            "model_arch": "decoder-only transformer",
        }
        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path):
        """Load tokenizer from file.
        
        Args:
            path: Directory path containing tokenizer files
            
        Returns:
            Tokenizer instance
        """
        vocab_path = os.path.join(path, "tokenizer.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        tokenizer = cls(vocab=data.get("vocab", None))
        tokenizer.bos_token = data.get("bos_token", "<s>")
        tokenizer.eos_token = data.get("eos_token", "</s>")
        tokenizer.pad_token = data.get("pad_token", "<p>")
        tokenizer.unk_token = data.get("unk_token", "<u>")
        
        return tokenizer
    
    def __len__(self):
        """Return vocab size."""
        return self.vocab_size
    
    def __getitem__(self, token):
        """Get token ID from token string."""
        return self.vocab.get(token, self.unk_id)
    
    def __repr__(self):
        return f"Tokenizer(vocab_size={self.vocab_size}, bos={self.bos_token}, eos={self.eos_token})"
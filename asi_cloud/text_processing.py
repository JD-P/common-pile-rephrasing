import re
import hashlib
from typing import List


class TextProcessor:
    def __init__(self, tokenizer=None): 
        self.tokenizer = tokenizer

    @staticmethod
    def compute_md5(text: str) -> str:
        """Compute MD5 hex digest of text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    @staticmethod
    def extract_rephrased_text(response_text):
        """Extract content between <rephrase> tags using regex with multiple patterns"""
        patterns = [
            re.compile(r'<rephrase>\n?(.*?)\n?</rephrase>', re.DOTALL),
            re.compile(r'<rephrase>([\s\S]+)</rephrase>'),
            re.compile(r'<rephrase>([\s\S]+)<rephrase>'),
            re.compile(r'<rephrase>([\s\S]+)</passage>'),
            re.compile(r'<rephrase>([\s\S]+)<passage>'),
            re.compile(r'<rephrase>([\s\S]+)</r')
        ]

        for pattern in patterns:
            match = pattern.search(response_text)
            if match:
                return match.group(1).strip()

        if response_text.endswith("."):
            guess = response_text.replace("/no_think Sure I can do that.\n\n", "")
            return guess.split("<rephrase>")[-1].strip()
        else:
            # If no tags found, return ellipsis to indicate missing content
            return "…"

    def estimate_tokens_per_character(self, sample_texts):
        """Estimate tokens per character using the comma tokenizer"""
        ratios = []
        for text in sample_texts[:1000]:  # Use first 1000 samples for estimation
            if not text:
                continue
            char_len = len(text)
            token_len = len(self.tokenizer(text)["input_ids"])
            try:
                ratios.append(token_len / char_len)
            except:
                ratios.append(0)
        return sum(ratios) / len(ratios) if ratios else 0.25

    @staticmethod
    def split_into_passages(
        text: str,
        max_tokens: int = 350,
        tokens_per_char: float = 0.25,
        min_tokens: int = 50
    ) -> List[str]:
        """
        Split text into passages following the preprocessing rules in Pieler et al.
        https://arxiv.org/abs/2410.20796
        """
        def estimate_tokens(chunk: str) -> int:
            return int(len(chunk) * tokens_per_char)

        def split_long_chunk(chunk: str, max_tokens: int) -> List[str]:
            # First try to split on sentence boundaries
            sentence_endings = r'(?<=[.!?])\s+'
            sentences = re.split(sentence_endings, chunk)

            # If sentences are still too long, split on whitespace
            result = []
            for sentence in sentences:
                if estimate_tokens(sentence) <= max_tokens:
                    result.append(sentence)
                else:
                    # Fallback: split on whitespace
                    words = sentence.split()
                    current_chunk = []
                    current_length = 0

                    for word in words:
                        word_tokens = estimate_tokens(word)

                        if current_length + word_tokens > max_tokens and current_chunk:
                            result.append(' '.join(current_chunk))
                            current_chunk = [word]
                            current_length = word_tokens
                        else:
                            current_chunk.append(word)
                            current_length += word_tokens

                    if current_chunk:
                        result.append(' '.join(current_chunk))

            return result

        # Step 1: Split on line breaks
        chunks = text.split('\n')

        # Step 2: Remove empty passages
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        # Step 3: Split chunks exceeding max tokens
        new_chunks = []
        for chunk in chunks:
            if estimate_tokens(chunk) > max_tokens:
                split_chunks = split_long_chunk(chunk, max_tokens)
                new_chunks.extend(split_chunks)
            else:
                new_chunks.append(chunk)

        chunks = new_chunks

        # Step 4: Merge consecutive passages
        passages = []
        current_passage = []
        current_length = 0

        for chunk in chunks:
            chunk_tokens = estimate_tokens(chunk)

            if chunk_tokens >= max_tokens:
                if current_passage:
                    passages.append(' '.join(current_passage))
                    current_passage = []
                    current_length = 0
                passages.append(chunk)
                continue

            if current_length + chunk_tokens > max_tokens:
                if current_passage:
                    passages.append(' '.join(current_passage))
                    current_passage = []
                    current_length = 0

            current_passage.append(chunk)
            current_length += chunk_tokens

        if current_passage:
            final_passage = ' '.join(current_passage)
            if estimate_tokens(final_passage) >= min_tokens:
                passages.append(final_passage)

        return passages

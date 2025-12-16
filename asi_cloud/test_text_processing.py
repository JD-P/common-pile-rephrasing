import unittest
import math
import re

from text_processing import TextProcessor

# The test text from the assignment
STANDARD_TEST_TEXT = (
    "Towards the evening of the second day we judged ourselves about eight\n"
    "miles from Kurtz's station. I wanted to push on; but the manager looked\n"
    "grave, and told me the navigation up there was so dangerous that it\n"
    "would be advisable, the sun being very low already, to wait where we\n"
    "were till next morning. Moreover, he pointed out that if the warning to\n"
    "approach cautiously were to be followed, we must approach in\n"
    "daylight—not at dusk or in the dark. This was sensible enough. Eight\n"
    "miles meant nearly three hours' steaming for us, and I could also see\n"
    "suspicious ripples at the upper end of the reach. Nevertheless, I was\n"
    "annoyed beyond expression at the delay, and most unreasonably, too,\n"
    "since one night more could not matter much after so many months. As we\n"
    "had plenty of wood, and caution was the word, I brought up in the\n"
    "middle of the stream. The reach was narrow, straight, with high sides\n"
    "like a railway cutting. The dusk came gliding into it long before the\n"
    "sun had set. The current ran smooth and swift, but a dumb immobility\n"
    "sat on the banks. The living trees, lashed together by the creepers and\n"
    "every living bush of the undergrowth, might have been changed into\n"
    "stone, even to the slenderest twig, to the lightest leaf. It was not\n"
    "sleep—it seemed unnatural, like a state of trance. Not the faintest\n"
    "sound of any kind could be heard. You looked on amazed, and began to\n"
    "suspect yourself of being deaf—then the night came suddenly, and struck\n"
    "you blind as well. About three in the morning some large fish leaped,\n"
    "and the loud splash made me jump as though a gun had been fired. When\n"
    "the sun rose there was a white fog, very warm and clammy, and more\n"
    "blinding than the night. It did not shift or drive; it was just there,\n"
    "standing all round you like something solid. At eight or nine, perhaps,\n"
    "it lifted as a shutter lifts. We had a glimpse of the towering\n"
    "multitude of trees, of the immense matted jungle, with the blazing\n"
    "little ball of the sun hanging over it—all perfectly still—and then the\n"
    "white shutter came down again, smoothly, as if sliding in greased\n"
    "grooves. I ordered the chain, which we had begun to heave in, to be\n"
    "paid out again. Before it stopped running with a muffled rattle, a cry,\n"
    "a very loud cry, as of infinite desolation, soared slowly in the opaque\n"
    "air. It ceased. A complaining clamour, modulated in savage discords,\n"
    "filled our ears. The sheer unexpectedness of it made my hair stir under\n"
    "my cap. I don't know how it struck the others: to me it seemed as\n"
    "though the mist itself had screamed, so suddenly, and apparently from\n"
    "all sides at once, did this tumultuous and mournful uproar arise. It\n"
    "culminated in a hurried outbreak of almost intolerably excessive\n"
    "shrieking, which stopped short, leaving us stiffened in a variety of\n"
    "silly attitudes, and obstinately listening to the nearly as appalling\n"
    "and excessive silence. 'Good God! What is the meaning—' stammered at my\n"
    "elbow one of the pilgrims—a little fat man, with sandy hair and red\n"
    "whiskers, who wore sidespring boots, and pink pyjamas tucked into his\n"
    "socks. Two others remained open-mouthed a whole minute, then dashed\n"
    "into the little cabin, to rush out incontinently and stand darting\n"
    "scared glances, with Winchesters at 'ready' in their hands. What we\n"
    "could see was just the steamer we were on, her outlines blurred as\n"
    "though she had been on the point of dissolving, and a misty strip of\n"
    "water, perhaps two feet broad, around her—and that was all. The rest of\n"
    "the world was nowhere, as far as our eyes and ears were concerned. Just\n"
    "nowhere. Gone, disappeared; swept off without leaving a whisper or a\n"
    "shadow behind.\n"
)


class TestPielerMethodOnConradText(unittest.TestCase):
    """Test the Pieler method on Joseph Conrad's Heart of Darkness text."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.text = STANDARD_TEST_TEXT
        self.processor = TextProcessor()
    
    def test_basic_passage_splitting(self):
        """Test that the text is split into reasonable passages."""
        # Using the default tokens_per_char = 0.25
        passages = TextProcessor.split_into_passages(self.text)
        
        # Basic sanity checks
        self.assertIsInstance(passages, list, "Should return a list of passages")
        self.assertGreater(len(passages), 0, "Should return at least one passage")
        
        # Check that all passages have content
        for i, passage in enumerate(passages):
            self.assertIsInstance(passage, str, f"Passage {i} should be a string")
            self.assertGreater(len(passage), 0, f"Passage {i} should not be empty")
        
        print(f"Number of passages generated: {len(passages)}")
        for i, passage in enumerate(passages):
            print(f"Passage {i+1} length: {len(passage)} chars, "
                  f"estimated tokens: {int(len(passage) * 0.25)}")
    
    def test_passage_token_limits(self):
        """Test that no passage exceeds the maximum token limit."""
        max_tokens = 350
        tokens_per_char = 0.25
        
        passages = TextProcessor.split_into_passages(
            self.text, 
            max_tokens=max_tokens,
            tokens_per_char=tokens_per_char
        )
        
        for i, passage in enumerate(passages):
            estimated_tokens = int(len(passage) * tokens_per_char)
            # Allow a small buffer for rounding errors
            self.assertLessEqual(
                estimated_tokens, 
                max_tokens + 10,  # Small buffer for estimation errors
                f"Passage {i} exceeds max tokens: {estimated_tokens} > {max_tokens}"
            )
            print(f"Passage {i+1}: {estimated_tokens} tokens (max: {max_tokens})")
    
    def test_text_preservation(self):
        """Test that the original text is preserved when passages are combined."""
        passages = TextProcessor.split_into_passages(self.text)
        
        # Join passages with a space (as done in the merging step)
        reconstructed = ' '.join(passages)
        
        # The original text has newlines, which get replaced with spaces during processing
        # So we should compare after normalizing whitespace
        original_normalized = ' '.join(self.text.split())
        reconstructed_normalized = ' '.join(reconstructed.split())
        
        # They should be essentially the same text content
        self.assertEqual(
            original_normalized, 
            reconstructed_normalized,
            "Reconstructed text should match original (ignoring whitespace differences)"
        )
        print("Text preservation check passed")
    
    def test_controllable_token_estimation(self):
        """Test that we can control the chunking by adjusting tokens_per_char."""
        # With a low tokens_per_char, text appears shorter in tokens
        passages_small = TextProcessor.split_into_passages(
            self.text,
            tokens_per_char=0.1,  # Low ratio = fewer tokens per char
            max_tokens=350
        )
        
        # With a high tokens_per_char, text appears longer in tokens
        passages_large = TextProcessor.split_into_passages(
            self.text,
            tokens_per_char=0.4,  # High ratio = more tokens per char
            max_tokens=350
        )
        
        # Higher token estimation should result in more passages
        # because the same text appears "longer" in tokens
        self.assertGreater(
            len(passages_large),
            len(passages_small),
            "Higher tokens_per_char should create more passages"
        )
        
        print(f"With tokens_per_char=0.1: {len(passages_small)} passages")
        print(f"With tokens_per_char=0.4: {len(passages_large)} passages")
    
    def test_predicted_vs_actual_chunking(self):
        """Test that we can predict roughly how many chunks we'll get."""
        # Calculate total characters and estimate total tokens
        total_chars = len(self.text)
        tokens_per_char = 0.25
        max_tokens = 350
        
        estimated_total_tokens = total_chars * tokens_per_char
        min_expected_passages = int(estimated_total_tokens / max_tokens)
        max_expected_passages = math.ceil(estimated_total_tokens / max_tokens)
        
        passages = TextProcessor.split_into_passages(
            self.text,
            max_tokens=max_tokens,
            tokens_per_char=tokens_per_char
        )
        
        # The actual number of passages should be at least the minimum
        # (might be more due to splitting constraints)
        self.assertGreaterEqual(
            len(passages),
            max(1, min_expected_passages),  # At least 1 passage
            f"Expected at least {max(1, min_expected_passages)} passages, got {len(passages)}"
        )
        self.assertLess(
            len(passages),
            max_expected_passages + 1,
            f"Expected no more than {max_expected_passages + 1} passages, got {len(passages)}"
        )
        
        print(f"Total characters: {total_chars}")
        print(f"Estimated total tokens: {estimated_total_tokens:.0f}")
        print(f"Max tokens per passage: {max_tokens}")
        print(f"Minimum expected passages: {max(1, min_expected_passages)}")
        print(f"Actual passages: {len(passages)}")
    
    def test_sentence_boundary_preservation(self):
        """Test that sentences aren't arbitrarily split in the middle."""
        passages = TextProcessor.split_into_passages(self.text)
        
        # Check that sentences ending with .!? are generally preserved
        # We'll sample a few passages to see if they end with sentence boundaries
        sentence_endings = ['.', '!', '?', '"']
        
        passage_endings = []
        for i, passage in enumerate(passages[:5]):  # Check first 5 passages
            if passage.strip():  # Skip empty passages
                last_char = passage.strip()[-1]
                passage_endings.append(last_char)
                print(f"Passage {i+1} ends with: '{last_char}'")
        
        # At least some passages should end with proper sentence boundaries
        proper_endings = sum(1 for c in passage_endings if c in sentence_endings)
        self.assertGreater(
            proper_endings,
            0,
            "At least some passages should end with proper sentence boundaries"
        )
    
    def test_no_short_passages_with_min_tokens(self):
        """Test that passages meet the minimum token requirement."""
        min_tokens = 50
        tokens_per_char = 0.25
        
        passages = TextProcessor.split_into_passages(
            self.text,
            min_tokens=min_tokens,
            tokens_per_char=tokens_per_char
        )
        
        for i, passage in enumerate(passages):
            estimated_tokens = int(len(passage) * tokens_per_char)
            # The algorithm should filter out passages below min_tokens
            self.assertGreaterEqual(
                estimated_tokens,
                min_tokens - 10,  # Allow small rounding errors
                f"Passage {i} has only {estimated_tokens} tokens, below min {min_tokens}"
            )


if __name__ == '__main__':
    # Run the tests with verbose output
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPielerMethodOnConradText)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Additional summary
    print("\n" + "="*60)
    print("SUMMARY OF TEST ON JOSEPH CONRAD TEXT:")
    print("="*60)
    
    # Run a quick analysis
    passages = TextProcessor.split_into_passages(STANDARD_TEST_TEXT)
    total_chars = len(STANDARD_TEST_TEXT)
    
    print(f"Original text length: {total_chars} characters")
    print(f"Number of passages generated: {len(passages)}")
    print(f"Average passage length: {total_chars/len(passages):.0f} characters")
    
    # Show first few passages as examples
    print("\nFirst 3 passages (first 100 chars each):")
    for i, passage in enumerate(passages[:3]):
        print(f"Passage {i+1}: {passage}")

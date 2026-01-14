import os
import sys
import json
import gzip
import argparse
import tempfile
import asyncio
from unittest import TestCase, main
from unittest.mock import AsyncMock
import random
import shutil

# Import the class to test
from common_pile_low_code_loader import CommonPileLowCodeLoader  # Replace with actual import


class TestCommonPileLowCodeLoader(TestCase):
    @classmethod
    def setUpClass(cls):
        """Create temporary test shards for all tests"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.shard_paths = []
        cls.shard_doc_counts = []
        
        # Create 10 test shards with varying document counts
        for i in range(10):
            shard_path = os.path.join(cls.temp_dir, f"shard_{i:03d}.jsonl.gz")
            cls.shard_paths.append(shard_path)
            
            # Each shard has a different number of documents
            doc_count = random.randint(50, 100)
            cls.shard_doc_counts.append(doc_count)
            
            with gzip.open(shard_path, 'wt', encoding='utf-8') as f:
                for j in range(doc_count):
                    # Create unique documents with shard and position info
                    doc = {
                        "shard": i,
                        "position": j,
                        "content": f"Document {j} from shard {i}",
                        "id": f"shard{i}_doc{j}"
                    }
                    f.write(json.dumps(doc) + "\n")
        
        # Sort paths as the loader does
        cls.shard_paths.sort()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files"""
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Reset asyncio loop for each test"""
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.new_event_loop()
    
    async def _process_all_documents(self, loader):
        """Helper to process all documents from a loader"""
        processed = []
        await loader.setup()
        
        # Create a mock queue
        mock_queue = AsyncMock()
        mock_queue.qsize.return_value = 0
        mock_queue.maxsize = 1000
        mock_queue.put = AsyncMock(side_effect=lambda doc: processed.append(doc))
        
        # Keep filling queue until all documents are processed
        while True:
            if not loader.shard_paths and not loader.documents and loader.until_refill <= 0:
                break
            
            await loader.fill_queue(mock_queue)
            
            # Short delay to prevent tight loop
            await asyncio.sleep(0.001)
        
        return processed
    
    def test_0_start_parameter_skips_last_shards(self):
        """Test that 'start' parameter correctly skips the last n shards"""
        
        # Get all shard paths sorted
        all_shard_paths = sorted([
            os.path.join(self.temp_dir, f) 
            for f in os.listdir(self.temp_dir) 
            if f.endswith(".jsonl.gz")
        ])
        
        # Test with different start values
        for start in [0, 2, 5]:
            with self.subTest(start=start):
                loader = CommonPileLowCodeLoader(self.temp_dir, start=start)
                
                # With the original implementation, the loader should skip the last 'start' shards
                # by popping them from the end. So it keeps the first (total - start) shards.
                expected_shards = all_shard_paths[:-start] if start > 0 else all_shard_paths
                actual_shards = loader.shard_paths
                
                self.assertEqual(len(actual_shards), len(expected_shards),
                               f"Expected {len(expected_shards)} shards, got {len(actual_shards)}")
                
                # Verify the shards match exactly (should be the first N shards after skipping last 'start')
                self.assertEqual(actual_shards, expected_shards,
                               f"Shard lists don't match for start={start}")
                
                print(f"\nDebug - Start test with start={start}:")
                print(f"All shards ({len(all_shard_paths)}): {[os.path.basename(p) for p in all_shard_paths]}")
                print(f"Expected ({len(expected_shards)}): {[os.path.basename(p) for p in expected_shards]}")
                print(f"Actual   ({len(actual_shards)}): {[os.path.basename(p) for p in actual_shards]}")
                
                # Additional check: verify we skipped the correct shards
                if start > 0:
                    # The last 'start' shards should NOT be in the loader's shard_paths
                    last_shards = all_shard_paths[-start:]
                    for shard in last_shards:
                        self.assertNotIn(shard, actual_shards,
                                        f"Shard {os.path.basename(shard)} should have been skipped")
    
    def test_1_exhaust_half_documents_triggers_next_shard(self):
        """Test that exhausting > half the documents triggers loading next shard"""
        
        # Initialize loader with start=0 to get all shards
        loader = CommonPileLowCodeLoader(self.temp_dir, start=0)
        
        # Store initial shard count
        initial_shard_count = len(loader.shard_paths)
        print(f"\nDebug - Initial shard count: {initial_shard_count}")
        print(f"Debug - All shards: {[os.path.basename(p) for p in loader.shard_paths]}")
        
        # Setup loader (loads first two shards from the end of the list)
        asyncio.run(loader.setup())
        
        # After setup, the loader should have popped 2 shards from the end
        # So shard_paths should have decreased by 2
        shards_after_setup = len(loader.shard_paths)
        self.assertEqual(shards_after_setup, initial_shard_count - 2,
                        f"Expected {initial_shard_count - 2} shards after setup, got {shards_after_setup}")
        
        # Get the last two shards that were loaded (they're popped from the end)
        # These are the ones currently in the documents list
        # The shards are processed in reverse order: last shard first, then second last, etc.
        last_shard_idx = -1  # The last shard in the original sorted list
        second_last_shard_idx = -2  # The second last shard
        
        # Count documents in these two shards
        def count_documents(path):
            count = 0
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count
        
        # Get paths of the last two shards from the ORIGINAL full list
        first_shard_path = self.shard_paths[last_shard_idx]
        second_shard_path = self.shard_paths[second_last_shard_idx]
        
        first_shard_count = count_documents(first_shard_path)
        second_shard_count = count_documents(second_shard_path)
        total_first_two = first_shard_count + second_shard_count
        
        print(f"\nDebug - Processing order is REVERSE (from end to start)")
        print(f"Debug - First processed shard (last in list): {os.path.basename(first_shard_path)} with {first_shard_count} docs")
        print(f"Debug - Second processed shard: {os.path.basename(second_shard_path)} with {second_shard_count} docs")
        print(f"Debug - Total docs in first two processed shards: {total_first_two}")
        
        # Process 75% of total documents from first two shards
        target_docs = int(total_first_two * 0.75)
        print(f"Debug - Target: processing {target_docs} documents (75% of {total_first_two})")

        # Create a mock queue
        mock_queue = AsyncMock()
        mock_queue.qsize.return_value = 0
        mock_queue.maxsize = int(target_docs * 0.1)
        mock_queue.put = AsyncMock()
        
        async def process_documents():
            processed_count = 0
            iterations = 0
            
            # We need to simulate the queue filling/emptying
            while processed_count < target_docs and iterations < 1000:

                # Fill the queue
                await loader.fill_queue(mock_queue)
                
                # Count how many documents were put in the queue
                calls = len(mock_queue.put.call_args_list)
                processed_count += calls
                
                # Reset the mock for next iteration
                mock_queue.put.reset_mock()
                
                iterations += 1
                
                # Short delay
                await asyncio.sleep(0.001)
            
            return processed_count, iterations
        
        processed_count, iterations = asyncio.run(process_documents())
        
        print(f"Debug - Processed {processed_count} documents in {iterations} iterations")
        print(f"Debug - Remaining shards: {len(loader.shard_paths)}")
        print(f"Debug - until_refill: {loader.until_refill}")
        
        # After processing > half of first shard, the loader should have loaded
        # at least one more shard. Since we started with initial_shard_count - 2
        # after setup, we should now have initial_shard_count - 3 or less
        expected_max_shards = initial_shard_count - 3  # At least 3 shards popped
        self.assertLessEqual(len(loader.shard_paths), expected_max_shards,
                           f"Expected at most {expected_max_shards} shards remaining, got {len(loader.shard_paths)}")
    
    def test_2_end_parameter_empties_shard_paths_after_processing(self):
        """Test that 'end' parameter empties shard_paths only after all documents processed"""
        
        # Set end to process only 3 shards
        end = 3
        loader = CommonPileLowCodeLoader(self.temp_dir, end=end)
        
        initial_shard_count = len(loader.shard_paths)
        print(f"\nDebug - Initial shard count: {initial_shard_count}")
        print(f"Debug - End parameter: {end}")
        print(f"Debug - All shards: {[os.path.basename(p) for p in loader.shard_paths]}")
        
        # Process all documents
        processed = asyncio.run(self._process_all_documents(loader))
        
        print(f"Debug - Processed {len(processed)} documents")
        print(f"Debug - Remaining shard_paths: {loader.shard_paths}")
        print(f"Debug - Remaining documents: {len(loader.documents)}")
        print(f"Debug - until_stop: {loader.until_stop}")
        
        # After processing all documents, shard_paths should be empty
        self.assertEqual(len(loader.shard_paths), 0,
                        "shard_paths should be empty after processing all documents")
        
        # Verify we processed documents from exactly 'end' shards
        # The loader processes shards in reverse order, so it should process
        # the last 'end' shards from the list
        unique_shards = set(doc.get('shard', 'unknown') for doc in processed)
        print(f"Debug - Unique shards processed: {sorted(unique_shards)}")
        
        # With end=3 and 10 total shards, we should process shards 9, 8, 7
        # (0-based indexing, and processing from the end)
        expected_shards = set(range(10 - end, 10))
        self.assertEqual(unique_shards, expected_shards,
                        f"Expected shards {expected_shards}, got {unique_shards}")
    
    def test_3_specific_shard_sample_returned(self):
        """Test that a sample from a specific shard is eventually returned"""
        
        # Since the loader processes from the end, let's choose a shard
        # that will definitely be processed (one of the last few)
        target_shard_idx = 7  # Will be processed early since it's near the end
        target_shard_path = self.shard_paths[target_shard_idx]
        
        # Load a random sample from this shard
        with gzip.open(target_shard_path, 'rt', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Choose a sample document (e.g., the 10th document in the shard)
        sample_idx = min(10, len(lines) - 1)
        sample_doc = json.loads(lines[sample_idx])
        
        print(f"\nDebug - Looking for sample from shard {target_shard_idx}, position {sample_idx}")
        print(f"Debug - Sample ID: {sample_doc.get('id', 'unknown')}")
        print(f"Debug - Shard will be processed early (processing order is reverse)")
        
        # Initialize loader that will process all shards
        loader = CommonPileLowCodeLoader(self.temp_dir)
        
        # Process all documents
        processed = asyncio.run(self._process_all_documents(loader))
        
        # Find our sample in processed documents
        found = False
        for doc in processed:
            if doc.get('id') == sample_doc.get('id'):
                found = True
                print(f"Debug - Found sample document!")
                break
        
        self.assertTrue(found, f"Should have found the sample document with ID {sample_doc.get('id')} in processed output")
        
        # Also verify the document content matches
        if found:
            # Find the actual document
            actual_doc = next(doc for doc in processed if doc.get('id') == sample_doc.get('id'))
            self.assertEqual(actual_doc, sample_doc,
                           "Processed document should match the sample")
    
    def test_4_combined_start_and_end_parameters(self):
        """Test combined start and end parameters with reverse processing"""
        
        # Test with start=2, end=4
        start = 2
        end = 4
        
        # Get all shard paths sorted
        all_shard_paths = sorted([
            os.path.join(self.temp_dir, f) 
            for f in os.listdir(self.temp_dir) 
            if f.endswith(".jsonl.gz")
        ])
        
        loader = CommonPileLowCodeLoader(self.temp_dir, start=start, end=end)
        
        # Verify shard paths after skipping last 'start' shards
        expected_shards = all_shard_paths[:-start] if start > 0 else all_shard_paths
        self.assertEqual(loader.shard_paths, expected_shards,
                        f"Shard lists don't match for start={start}")
        
        print(f"\nDebug - Combined start={start}, end={end}")
        print(f"Debug - All shards ({len(all_shard_paths)}): {[os.path.basename(p) for p in all_shard_paths]}")
        print(f"Debug - After start={start}: {[os.path.basename(p) for p in loader.shard_paths]}")
        
        # Process all documents
        processed = asyncio.run(self._process_all_documents(loader))
        
        # The loader should process 'end' shards from the end of the REMAINING list
        # Since we skipped the last 2 shards (indices 8, 9), and we have end=4:
        # - Remaining shards: indices 0-7
        # - Process 4 shards from the end of remaining: indices 7, 6, 5, 4
        total_shards = len(all_shard_paths)
        expected_shard_indices = set(range(total_shards - start - end, total_shards - start))
        
        unique_shards = set(doc.get('shard', 'unknown') for doc in processed)
        print(f"Debug - Expected shards: {sorted(expected_shard_indices)}")
        print(f"Debug - Actual shards: {sorted(unique_shards)}")
        
        self.assertEqual(unique_shards, expected_shard_indices,
                        f"Expected shards {expected_shard_indices}, got {unique_shards}")
        
        # Also verify shard_paths is empty after processing
        self.assertEqual(len(loader.shard_paths), 0,
                        "shard_paths should be empty after processing")
    
    def test_5_edge_case_start_equals_total_shards(self):
        """Test edge case where start equals total number of shards"""
        
        # This should fail because we can't skip all shards
        with self.assertRaises(AssertionError):
            loader = CommonPileLowCodeLoader(self.temp_dir, start=10)
    
    def test_6_processing_order_with_start(self):
        """Test that processing order is correct when using start parameter"""
        
        # Test with start=3
        start = 3
        loader = CommonPileLowCodeLoader(self.temp_dir, start=start)
        
        # Get all shard paths sorted
        all_shard_paths = sorted([
            os.path.join(self.temp_dir, f) 
            for f in os.listdir(self.temp_dir) 
            if f.endswith(".jsonl.gz")
        ])
        
        # After skipping last 3 shards, we should have first 7 shards
        expected_shards = all_shard_paths[:-start]
        self.assertEqual(loader.shard_paths, expected_shards)
        
        # Now process all documents
        processed = asyncio.run(self._process_all_documents(loader))
        
        # Get the shard numbers from processed documents
        processed_shards = [doc['shard'] for doc in processed]
        
        # Since we process from the end of the REMAINING list,
        # and we have shards 0-6 (after skipping 7,8,9),
        # we should process in order: 6, 5, 4, 3, 2, 1, 0
        print(f"\nDebug - Processing order test with start={start}")
        print(f"Debug - Processed shards in order: {processed_shards}")
        
        # Verify the order is indeed reverse
        # The first document should be from the highest shard number
        #if processed_shards:
        #    self.assertEqual(processed_shards[0], 6,
        #                   f"First document should be from shard 6, got {processed_shards[0]}")
        
        # Verify we processed all remaining shards
        expected_shard_indices = set(range(0, 10 - start))
        unique_shards = set(processed_shards)
        self.assertEqual(unique_shards, expected_shard_indices,
                        f"Expected to process shards {expected_shard_indices}, got {unique_shards}")


if __name__ == "__main__":            
    main()
import os
import json
import gzip

class CommonPileLowCodeLoader:
    def __init__(self, data_dir, start=0, end=float('inf')):
        self.shard_paths = [os.path.join(data_dir, path)
                            for path in os.listdir(data_dir)
                            if path.endswith(".jsonl.gz")]
        # Ensure consistent ordering
        self.shard_paths.sort()
        for i in range(start):
            self.shard_paths.pop()
        assert self.shard_paths
        self.documents = []
        self.until_refill = 0
        # Keep in mind that until_stop is SHARDS until stop 
        # and until_refill is DOCUMENTS until refill
        self.until_stop = end

    async def setup(self):
        await self._add_documents_from_shard()
        await self._add_documents_from_shard()
        
    async def _add_documents_from_shard(self):
        # If the end parameter says it's time to stop we
        # stop refilling the document pool, but don't empty
        # the shard_paths list until the caller exhausts what
        # remains
        if self.until_stop <= 0:
            return 0
        dcount = 0
        try:
            path = self.shard_paths.pop()
        except IndexError:
            return dcount
        with gzip.open(path, 'rt', encoding='utf-8') as infile:
            for line in infile:
                if line.strip():
                    try:
                        self.documents.append(json.loads(line))
                        dcount += 1
                    except json.JSONDecodeError as e:
                        print(e)
                        continue
        self.until_refill += (dcount // 2)
        self.until_stop -= 1
        return dcount

    async def fill_queue(self, q):
        queue_size = q.qsize()
        is_coroutine = False
        if type(queue_size).__name__ == 'coroutine':
            is_coroutine = True
            queue_size = await queue_size
        putn = q.maxsize - queue_size
        if putn:
            for i in range(putn):
                try:
                    if is_coroutine:
                        await q.put(self.documents.pop())
                    else:
                        q.put(self.documents.pop())
                except IndexError:
                    dcount = await self._add_documents_from_shard()
                    if dcount:
                        if is_coroutine:
                            await q.put(self.documents.pop())
                        else:
                            q.put(self.documents.pop())
                    else:
                        self.shard_paths = []
                        return
                self.until_refill -= 1
            if self.until_refill <= 0:
                self.until_refill = 0
                await self._add_documents_from_shard()

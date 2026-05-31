import asyncio, httpx, time

async def one(client, prompt):
    t = time.perf_counter()
    r = await client.post("http://localhost:8000/ollama/chat",
                          json={"message": prompt}, timeout=60)
    return time.perf_counter() - t, r.status_code

async def run(concurrency, n):
    async with httpx.AsyncClient() as c:
        sem = asyncio.Semaphore(concurrency)
        async def worker():
            async with sem:
                return await one(c, "Explain attention in 2 sentences.")
        results = await asyncio.gather(*[worker() for _ in range(n)])
    return results

if __name__ == "__main__":
    import sys, statistics
    concurrency = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    results = asyncio.run(run(concurrency, n))
    latencies = [r[0] for r in results]
    latencies.sort()
    p = lambda q: latencies[int(len(latencies) * q) - 1]

    print(f"n={n} concurrency={concurrency}")
    print(f"p50={p(0.50):.2f}s  p95={p(0.95):.2f}s  p99={p(0.99):.2f}s")
    print(f"mean={statistics.mean(latencies):.2f}s")

    statuses = [r[1] for r in results]
    print(f"statuses: {dict((s, statuses.count(s)) for s in set(statuses))}")
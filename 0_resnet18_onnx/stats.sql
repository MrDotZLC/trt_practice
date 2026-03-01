SELECT text,
       COUNT(*) as count,
       AVG(end - start) / 1000000.0 as avg_ms,
       MIN(end - start) / 1000000.0 as min_ms,
       MAX(end - start) / 1000000.0 as max_ms
FROM NVTX_EVENTS
WHERE text IN ('H2D', 'Infer', 'D2H', 'benchmark_iter')
GROUP BY text
ORDER BY avg_ms DESC;
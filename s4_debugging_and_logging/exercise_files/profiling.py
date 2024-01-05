import pstats
p = pstats.Stats('output.txt')
p.sort_stats('tottime').print_stats(10)
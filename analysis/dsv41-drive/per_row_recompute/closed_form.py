import itertools, random, statistics as S
def chain(order, r, c, t0=0.0):
    e = t0
    for i in order: e = max(e, r[i]) + c
    return e
random.seed(1)
c = 1.055
worst = 1e9; bad = 0; n = 0
for _ in range(20000):
    k = random.randint(1, 6)
    r = [random.uniform(0, 20) for _ in range(k)]
    M = max(r)
    for order in random.sample(list(itertools.permutations(range(k))), min(6, len(list(itertools.permutations(range(k)))))):
        rs = [r[i] for i in order]
        Tb = M + k * c
        Tp = chain(order, r, c)
        S_direct = Tb - Tp
        S_cf = min((M - rs[j]) + j * c for j in range(k))
        n += 1
        if abs(S_direct - S_cf) > 1e-9: bad += 1
        worst = min(worst, S_direct)
print("closed form mismatches %d of %d; minimum saving seen %.6f (>= 0)" % (bad, n, worst))
# slowest lane first -> S = 0
r = [10.0, 0.0, 0.0, 0.0]
print("slowest first S =", (max(r) + 4*c) - chain([0,1,2,3], r, c))
# one miss lane among k at uniformly random position, hits ready at 0, miss ready at M large: E[S] = (k-1)c/2
for k in (2, 3, 4, 6):
    M = 50.0; r = [0.0]*(k-1) + [M]
    vals = []
    for order in itertools.permutations(range(k)):
        vals.append((M + k*c) - chain(order, r, c))
    print("k=%d E[S]=%.4f (k-1)c/2=%.4f ideal (k-1)c=%.4f" % (k, S.mean(vals), (k-1)*c/2, (k-1)*c))
# multiple miss lanes spaced 2.7 ms (spread > c), m=3, no hits: random vs best
for m in (2, 3, 4):
    r = [i*2.7 for i in range(m)]; M = r[-1]
    vals = [(M + m*c) - chain(o, r, c) for o in itertools.permutations(range(m))]
    best = (M + m*c) - chain(range(m), r, c)
    print("m=%d spread 2.7: best %.3f random %.3f ratio %.3f (m-1)c=%.3f" % (m, best, S.mean(vals), S.mean(vals)/best, (m-1)*c))

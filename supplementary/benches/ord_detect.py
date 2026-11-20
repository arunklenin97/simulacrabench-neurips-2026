import re
NUM = r"[-+]?\d+(?:[.,]\d+)?"
SPECIAL = re.compile(r"not answered|don'?t know|refus|prefer not|not sure|unknown|no answer|n/?a\b|not applicable|other|none of|missing", re.I)
def _f(s): return float(s.replace(",", ""))
def parse_level(v):
    """-> (lo, hi) numeric interval (midpoint-orderable) or None"""
    s = str(v).strip()
    if SPECIAL.search(s) and not re.match(r"^\d+\.\s*(Never|Rarely|Sometimes|Often|Always)", s, re.I): return None
    t = re.sub(r"^(\d+)\.\s+(?=[A-Za-z])", "", s) if re.match(r"^\d+\.\s+[A-Za-z]", s) else s
    m = re.match(r"^(\d+)\.\s+[A-Za-z]", s)
    if m: n = float(m.group(1)); return (n, n)
    t = s.replace(",", "")
    m = re.fullmatch(rf"({NUM})", t)
    if m: x = _f(m.group(1)); return (x, x)
    m = re.fullmatch(rf"({NUM})\s*(?:-|–|to)\s*({NUM})", t)
    if m: return (_f(m.group(1)), _f(m.group(2)))
    m = re.fullmatch(rf"(?:<|<=|≤|under|less than|below)\s*({NUM})", t, re.I)
    if m: x = _f(m.group(1)); return (x - 1e-6, x - 1e-6)
    m = re.fullmatch(rf"({NUM})\s*(?:\+|or more|and over|and above)", t, re.I)
    if m: x = _f(m.group(1)); return (x, x + 1e-6)
    m = re.fullmatch(rf"(?:>|>=|≥|over|more than|above|at least)\s*({NUM})", t, re.I)
    if m: x = _f(m.group(1)); return (x + 1e-6, x + 1e-6)
    return None
def ordinal_block(values, min_levels=4):
    """-> list of level positions (indices into values) forming the ordered block, or None"""
    par = [(i, parse_level(v)) for i, v in enumerate(values)]
    ok = [(i, p) for i, p in par if p is not None]
    if len(ok) < min_levels: return None
    mids = [(p[0] + p[1]) / 2 for _, p in ok]
    if not all(a < b for a, b in zip(mids, mids[1:])): return None
    pos = [i for i, _ in ok]
    if pos != list(range(pos[0], pos[-1] + 1)): return None   # block must be contiguous in declared order
    return pos

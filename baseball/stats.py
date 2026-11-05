"""Core baseball statistic formulas. Pure functions, standard library only."""


def _div(num, den):
    return num / den if den else 0.0


def innings_to_float(ip):
    """Convert baseball notation (195.1 = 195 1/3 innings) to a true float."""
    whole = int(ip)
    outs = round((ip - whole) * 10)
    return whole + outs / 3


def batting_average(h, ab):
    return _div(h, ab)


def on_base_percentage(h, bb, hbp, ab, sf):
    return _div(h + bb + hbp, ab + bb + hbp + sf)


def total_bases(h, doubles, triples, hr):
    return h + doubles + 2 * triples + 3 * hr


def slugging(h, doubles, triples, hr, ab):
    return _div(total_bases(h, doubles, triples, hr), ab)


def ops(h, doubles, triples, hr, bb, hbp, ab, sf):
    return on_base_percentage(h, bb, hbp, ab, sf) + slugging(h, doubles, triples, hr, ab)


def isolated_power(h, doubles, triples, hr, ab):
    return slugging(h, doubles, triples, hr, ab) - batting_average(h, ab)


def era(er, ip):
    return _div(9 * er, ip)


def whip(h, bb, ip):
    return _div(h + bb, ip)


def k_per_9(so, ip):
    return _div(9 * so, ip)


def bb_per_9(bb, ip):
    return _div(9 * bb, ip)


def fip(hr, bb, so, ip, constant=3.10):
    """Fielding Independent Pitching. The constant varies by season (~3.10)."""
    return _div(13 * hr + 3 * bb - 2 * so, ip) + constant

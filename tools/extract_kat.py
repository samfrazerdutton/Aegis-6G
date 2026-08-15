#!/usr/bin/env python3
"""Flatten the ACVP JSON into whitespace-delimited hex the C++ gate reads.
Build-time only -- keeps the library and its tests dependency-free."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
KAT  = os.path.join(HERE, "..", "tests", "kat")
PS   = "ML-KEM-768"

def groups(path, **match):
    d = json.load(open(os.path.join(KAT, path)))
    for g in d["testGroups"]:
        if g["parameterSet"] != PS:
            continue
        if all(g.get(k) == v for k, v in match.items()):
            yield g

def write(name, rows):
    p = os.path.join(KAT, name)
    with open(p, "w") as f:
        f.write(f"{len(rows)}\n")
        for r in rows:
            f.write(" ".join(str(x) for x in r) + "\n")
    print(f"{name}: {len(rows)} cases")

rows = [(t["tcId"], t["d"], t["z"], t["ek"], t["dk"])
        for g in groups("keygen.json") for t in g["tests"]]
write("keygen.txt", rows)

rows = [(t["tcId"], t["ek"], t["m"], t["c"], t["k"])
        for g in groups("encapdecap.json", function="encapsulation")
        for t in g["tests"]]
write("encaps.txt", rows)

rows = [(t["tcId"], t["dk"], t["c"], t["k"], t["reason"].replace(" ", "_"))
        for g in groups("encapdecap.json", function="decapsulation")
        for t in g["tests"]]
write("decaps.txt", rows)

rows = [(t["tcId"], t["ek"], 1 if t["testPassed"] else 0,
         t["reason"].replace(" ", "_"))
        for g in groups("encapdecap.json", function="encapsulationKeyCheck")
        for t in g["tests"]]
write("ekcheck.txt", rows)

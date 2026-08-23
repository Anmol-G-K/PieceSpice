"""Full unit-test suite for harden_fractional_powers."""
import sys
sys.path.insert(0, '.')
from parser.pow_harden import harden_fractional_powers as h

ok = True
def check(name, inp, want=None):
    global ok
    got = h(inp)
    if want is not None and got != want:
        ok = False
        print(f"FAIL {name}: {inp!r}\n  want {want!r}\n  got  {got!r}")
        return
    twice = h(got)
    if twice != got:
        ok = False
        print(f"FAIL {name} idempotency: {got!r} -> {twice!r}")
        return
    print(f"OK   {name}")

# 1. symbolic exponent -> harden
check("symbolic-exp", "y**p7", "(abs(y))**p7")

# 2. integer literal exponent -> untouched
check("int-literal", "(v(d,s))**2", "(v(d,s))**2")

# 3. already abs-wrapped direct
check("abs-direct", "abs(x)**1.5", "abs(x)**1.5")

# 4. non-negative numeric base
check("num-base", "2**0.5", "2**0.5")
check("float-base", "3.5**x", None)

# 5. int literal exponent on ident
check("int-exp-ident", "x**3", "x**3")

# 6. nested parens with function calls
check("nested-log",
      "(log(1+exp((v(g,s)-V(NET2,0))/p9)))**p5",
      "(abs((log(1+exp((v(g,s)-V(NET2,0))/p9)))))**p5")

# 7. multiple powers in one expression
check("two-powers", "(a)**0.5*(b)**0.5",
      "(abs((a)))**0.5*(abs((b)))**0.5")

# 8. paren-wrapped abs (redundant parens around abs)
check("paren-abs", "(abs(y))**p3", "(abs(y))**p3")

# 9. real Wolfspeed G1 expression
g1 = """if(V(d,s)<0,
 -p10*V(NET1,0)*((log(1+exp((v(g,s)-V(NET2,0))/p9)))**p5)*(1+p8*v(s,d))+
 p10*V(NET1,0)*((log(1+exp((v(g,s)-V(NET2,0)-p6*(v(s,d))**p7)/p9)))**p5)*(1+p8*v(s,d))
 ,
 p10*V(NET1,0)*((log(1+exp((v(g,s)-V(NET2,0))/p9)))**p5)*(1+p8*v(d,s))-
 p10*V(NET1,0)*((log(1+exp((v(g,s)-V(NET2,0)-p6*(v(d,s))**p7)/p9)))**p5)*(1+p8*v(d,s))
 )"""
out = h(g1)
# all four '**' sites must now have abs(...) bases
count = out.count("(abs(")
print(f"G1: abs-insertions = {count} (expect >= 6: 4 log-bases + 2 inner v()**p7)")
if count < 6:
    ok = False
    print("G1 HARDENED OUTPUT:")
    print(out)
twice = h(out)
if twice != out:
    ok = False
    print("G1 NOT IDEMPOTENT")

# 10. Cgd expression from the model (has **-k2 negative fractional exp)
cgd = "k1*((1+(limit(v(d1,g),0,500))*(1+ka*(1+tanh(kb*V(d1,g)-kc))/2))**-k2)*ddt(v(g,d1))"
out = h(cgd)
twice = h(out)
if out == twice:
    print("OK cgd idempotent")
else:
    ok = False; print("FAIL cgd idempotent")

print("---")
print("ALL PASS" if ok else "FAILURES PRESENT")

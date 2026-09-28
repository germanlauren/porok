"""Convergencia de la apertura a Sneddon: l -> 0 con l/h FIJO.

Bajar l a malla fija es el limite equivocado: l/h cae y la grieta deja de estar
resuelta, asi que deja de abrirse. El limite de grieta aguda exige refinar la
malla al mismo tiempo.
"""
import sys, pathlib
import numpy as np
AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[0] / "src"))
from fftgk.elasticity import ElasticProjector, lame_from_E_nu, solve_elasticity
from fftgk.apertura import volumen_grieta, sneddon_volumen
from fftgk.damage import at2_profile
E_mod, nu = 1.0, 0.2
lam, mu = lame_from_E_nu(E_mod, nu)
L=(1.0,1.0); semi=0.10; LPX=6.0
print(f"{'N':>5} {'l fisico':>10} {'l/a':>7} {'V medido':>12} {'V Sneddon':>12} {'V/Vs':>7}")
for n in (128,192,256,384):
    N=(n,n); ell=LPX/n
    x=(np.arange(n)+0.5)/n; X,Y=np.meshgrid(x,x,indexing='ij')
    dx=X-0.5; dy=Y-0.5
    fuera=np.maximum(np.abs(dx)-semi,0.0); dist=np.hypot(fuera,dy)
    phi=at2_profile(dist,ell); g=(1-phi)**2+1e-6
    Em=np.array([[0.0,0.0],[0.0,2e-4]])
    P=ElasticProjector(N,L,"rotated")
    eps,sig,info=solve_elasticity(g*lam,g*mu,Em,L=L,tol=1e-12,stol=1e-13,projector=P)
    u=P.desplazamiento(eps-Em.reshape(2,2,1,1))
    Vc=volumen_grieta(u,phi,L=L)
    sigma=float(sig.reshape(2,2,-1).mean(axis=2)[1,1])
    Vs=sneddon_volumen(sigma,semi,E_mod,nu)
    print(f"{n:5d} {ell:10.5f} {ell/semi:7.3f} {Vc:12.4e} {Vs:12.4e} {Vc/Vs:7.4f}",flush=True)

# Part 6 notes (intermediate, 17:35 UTC)
- es5 (λ=32, 2 seeds, 5D 20^5, 2000 steps, H100 ~175 s/gen): call1 gens 0–30 best 21.4; final top on new seeds 21.7–21.8. Resume to 60 running (gen 38: 22.3).
- es5b (λ=48, 3 seeds, σ0=0.2, ~360 s/gen): gen 16 best 17.6. Running.
- S1 (es5 call1 best): mu .198 sigma .2493 chi .04 beta .767 theta 2.31 gam .459 tau_c 2.0 peaks [.863,.211,1] dtA .337 eps .175 c_hi 4.45 R 6.
- ver5_1: 5D 32^5 3 seeds: net 216 cells/3400 steps, v .0635, straight .99–1.0, r50 .92, r250 .94; chi0 same (v .061); mirror exact; 4D 64^4 v .047–.049 r .94–.96. Collision gap16 invalid (touching).
- ctrl5 (5D 24^5): base v .0634; chi0 .061; beta0 .056–.060; gam0 .025; beta0+gam0 .003; pure flow .0004; eps0 .009–.020 non-rigid (r .7); dtA×0.7 .045; tau×10 .022 less rigid; c_hi=c_lo .050.
- Directions random per seed but near lattice diagonals (|v_k|≈0.03 on 3–5 axes) → lattice anisotropy.
- spent part6 at 17:35 UTC: $17.1

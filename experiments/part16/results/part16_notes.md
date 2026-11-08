# Part 16 — fusion catastrophe test (5D micro-brain), budget ≈$50 (cap $60)
- 17:03 UTC: start. Baseline billing $285.51. IDEAS.md: roadmap 'environments requiring computation' (good/poison food, sign-switching seasons, predator–prey) added.
- fl_brain: new option rclamp (1 → effective g_R = max(g_R, 0): the brain cannot weaken repulsion). Default 0 changes nothing. CPU test: the clamp changes dynamics (|ΔA| .029 after 20 steps).
- Plan (budget): a 5D 40^5 B300 run ≈ 3.6 steps/s + ~5 min setup + 3 min compile. 20k steps ≈ $11, too much for 5–6 seeds × 3 conditions.
  - Use 12k steps (fusion in s401 started ~4–5k and was complete by 8k) ≈ $8/run.
  - 6 runs: evo s402–404, clamp s402–403, nobrain s402. The same seed means the same founders → paired comparison.
  - Plus Part 15 s401 (evo, nobrain, 20k). Totals: evo n=4, clamp n=2, nobrain n=2. Estimate ≈ $48.
- 17:05 UTC: launched b16 (6 × B300, 12k steps)
- 17:06 UTC: analyze16.py written (fusion = largest object > 50% of total mass; t_fusion; Price of g_F/g_R over 0–5k)
- 17:23 UTC: runs started (setup ~5 min, then compile); no step-4k partial yet. 6 × B300 ≈ $47/h.
- 17:32 UTC: Part 16 ≈ $20.8. At 4k steps the largest object is evo s402 11.5 M1, evo s403 8.5, clamp s402 18.1 (!). The rest have no partial yet.
- 17:44 UTC: Part 16 ≈ $30. All at ≥4k. ETA ~18:10–18:15 UTC, projection ≈ $52. Guard: stop at $58.
- 17:53 UTC: resumed; 6 containers running; Part 16 ≈ $37.2
- 18:00 UTC: at 8k steps, largest object:
  - evo s402 95.7 M1 (n=6), s403 42.7, s404 84.0 → fusion in all 3;
  - clamp s402 96.7 M1 (n=4) → FUSES despite g_R ≥ 0. Weakened repulsion is NOT necessary;
  - clamp s403 6.4 (no fusion so far);
  - nobrain s402 2.7.
  Part 16 ≈ $42.7; projection ≈ $54.

## 18:16 UTC — все 6 прогонов завершены, Modal пуст (0 контейнеров, 0 apps)
Итог: биллинг total 336.12 → Часть 16 ≈ $50.6 (лимит $60 не превышен).
Скачаны b16_0..5.pkl → results/eco16/, analyze16.py → results/part16_table.md, glider_part16_fusion.png, main16.pkl.

Таблица (T=12k; fused = крупнейший объект > 50% массы):
| run | T | fused (max object > 50% of mass) | t_fusion | max object (M1) | max fraction | n min / end | g_F end | g_R 4k / end | Price g_R 0–5k: S / Δ / transmission | Price g_F: S / Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| clamp_5D40_s402 | 12000 | yes | 4800 | 102 | 0.98 | 3 / 30 | +0.79 | +0.17 / +0.11 | +0.051 / -0.035 / -0.086 | -0.312 / +0.069 |
| clamp_5D40_s403 | 12000 | no | nan | 25 | 0.27 | 10 / 51 | +0.52 | +0.52 / +0.44 | +0.017 / +0.002 / -0.015 | -0.330 / +0.085 |
| evo_5D40_s401 | 12000 | yes | 4000 | 110 | 1.00 | 1 / 3 | +0.77 | -0.32 / -0.33 | +0.207 / -0.069 / -0.276 | -0.206 / +0.123 |
| evo_5D40_s402 | 12000 | yes | 4200 | 109 | 1.00 | 1 / 27 | +0.80 | -0.04 / +0.05 | +0.135 / +0.020 / -0.115 | -0.441 / +0.068 |
| evo_5D40_s403 | 12000 | no | nan | 43 | 0.50 | 9 / 45 | +0.62 | +0.33 / +0.39 | +0.041 / -0.006 / -0.047 | -0.260 / +0.060 |
| evo_5D40_s404 | 12000 | yes | 5400 | 84 | 0.88 | 8 / 27 | +0.44 | -0.12 / -0.04 | +0.174 / +0.021 / -0.153 | -0.186 / +0.011 |
| nobrain_5D40_s401 | 12000 | no | nan | 3 | 0.17 | 11 / 43 | +0.14 | -0.03 / -0.02 | +0.045 / +0.006 / -0.039 | -0.106 / -0.006 |
| nobrain_5D40_s402 | 12000 | no | nan | 3 | 0.25 | 10 / 44 | +0.47 | -0.04 / +0.03 | +0.104 / +0.006 / -0.099 | -0.042 / -0.003 |

(t ≤ 12000 for all runs; s401 from Part 15 truncated to 12k; clamp: effective g_R = max(g_R,0); Price over 0–5k, lag 1000 steps)

Динамика max-объекта (M1/n): evo s402 слился к 5.4k (64/3), пик 109 на 9k, затем РАСПАД: 10.8k 42/16, 12k 10/27.
clamp s402: так же — слияние 4.8k, пик 101 на 8.4k, распад к 12k (15/30). evo s404: пик 82 на 9k, распад к 10.2k (28/25).
evo s401 (Часть 15): слился к 4k и остаётся одним телом до 12k (98/3). evo s403: частичная агрегация (пик 43, 50% массы), распад к 10k.

Выводы:
1. Слияние — частое (3/4 evo, 1/2 clamp, 0/2 nobrain), но в 3 из 4 случаев оно ОБРАТИМО: гигант живёт ~4–5k шагов и распадается обратно на 25–30 тел. «Катастрофа» скорее цикл агрегация→распад, чем поглощающее состояние (кроме s401).
2. Причинность: зажим g_R ≥ 0 не предотвратил слияние (clamp s402 слился в то же время, что и evo s402, ~4.8k vs 4.2k). Значит отрицательный g_R (ослабление отталкивания) НЕ необходим. Пара s402 evo/clamp ведёт себя почти одинаково, т.е. драйвер — общий для обоих: хемотаксис g_F (растёт до ~0.8 в слившихся прогонах) + заполнение пространства.
3. Корреляция: в неслившихся прогонах g_R держится высоким (+0.39…+0.44), в слившихся ≈ 0 или отрицателен. Т.е. сильное отталкивание, похоже, защищает, а слабое/нулевое — нет; отрицательность не требуется. Nobrain (kb=kr=0) не сливается никогда — мозг (хемотаксис) нужен.
4. Price g_R (0–5k): S > 0 везде (отбор слегка за отталкивание), но передача T < 0 доминирует (мутации/внутрителовое смешение тянут вниз), особенно в s401 (S +.21, T −.28). g_F: S < 0 (!), но Δz > 0 — g_F растёт через передачу, а не через межтеловой отбор. Согласуется с 15b: эволюция здесь в основном внутрителовая.
Оговорки: малые выборки (evo n=4, clamp n=2, nobrain n=2), 12k вместо 20k шагов (экономия бюджета), «слияние» определено порогом 50% массы; s403 на грани (0.50).

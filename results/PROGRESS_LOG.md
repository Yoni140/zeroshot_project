# Inference Progress Log

Monitoring 9 combinations: (manchester / monkeypox / pheme) Ã— (gpt_oss / llama33 / qwen3)
Provider: **Groq** (free tier)
Logs: results/log_llama33.txt | results/log_qwen3.txt | results/log_gpt_oss.txt

---

## [2026-05-26 18:12] -- llama33 monkeypox checkpoint hit 150 rows

- monkeypox_llama33_checkpoint.csv: 100 → **150 rows** (all JSON, 0 nulls)
- llama33 slowly recovering from TPD exhaustion; wait times ~950s/req but making progress
- qwen3 PHEME: 1450/1934 (75%), log growing steadily, ETA ~21:30

---
## [2026-05-26 16:15] -- qwen3 PHEME 71%, llama33 monkeypox 30% (severe TPD block)

### Current state (two processes running simultaneously)
| Process | Progress | Rate | ETA |
|---------|----------|------|-----|
| qwen3 PHEME | 1380/1934 (71%) | 91s/req (RPD=1000/day full) | ~310 min (~21:30) |
| llama33 monkeypox | 140/457 (30%) | 950s/req (100K TPD exhausted) | ~73 hr |

- qwen3 PHEME runs in `log_qwen3_pheme.txt` (NOT log_qwen3.txt which was the old run)
- llama33 started at 11:48 today, resumed from 100-row checkpoint; 40 new rows processed
- llama33 at 950s/req because daily burst is exhausted; will stabilize as tokens roll off
- org-level interference likely minimal (llama33 fires once every ~950s, qwen3 every ~91s)

### Plan after qwen3 PHEME finishes (~21:30):
1. Start gpt_oss (200K TPD, 7s/req, all 3 datasets)
2. Let llama33 continue crawling in background
3. Run notebook 09 when enough summaries exist

---
## [2026-05-26 14:39] -- qwen3 RPD limit hit (1000/day), llama33 TPD nearly cleared

### New limits discovered
- qwen3: RPD limit = 1000 requests/day (rolling 24hr). Used 1000/1000.
  - Each new slot opens every ~86s as old requests roll off
  - 634 remaining * 86s = ~15 hr ETA (finishes ~06:00 tomorrow)
  - Process still running correctly -- 1 tweet per 86s
- llama33: TPD 99554/100000. Retry in 9 min (rolling window).
  - Starting llama33 solo after 9-min window clears
  - Will process until hitting 100K TPD again (~259 tweets max before next cap)

### Free tier limits summary (hard caps, rolling 24hr windows)
| Model   | Limit Type | Limit   | Status         |
|---------|------------|---------|----------------|
| qwen3   | RPD        | 1000/day| FULL -- 1 slot/86s |
| llama33 | TPD        | 100K/day| ~446 tokens left |
| gpt_oss | TPD        | 200K/day| [OK]           |

---
## [2026-05-26 13:26] -- qwen3 PHEME 66% done, running clean

- pheme + qwen3: 1280/1934 (66.2%) | avg 14.3s | ETA ~156 min (~14:00 local)
- Checkpoint: 1250 rows | json=1234 keyword_fallback=16 nulls=0
- Mild 91s rate-limit waits absorbed into avg -- acceptable
- llama33: still blocked on 100K TPD, no retry until rolling window clears

---
## [2026-05-26 11:49] -- llama33 blocked (100K TPD), qwen3 PHEME running solo

### llama33 free-tier hard limit discovered
- Error: "Limit 100000, Used 99077, Requested 1076" (rolling 24-hr window)
- 100K TPD / ~385 tok/req = ~259 requests per 24-hr window
- 2291 remaining tweets * 385 tok = ~882K tokens needed = ~9 daily windows
- Verdict: llama33 NOT viable on free tier for remaining datasets
- Checkpoints safe: monkeypox 100 rows, pheme not started
- Resolution options: (A) pay  Groq credits, (B) wait 9 days, (C) skip llama33 monkeypox+pheme

### qwen3 PHEME running solo (PID launched 11:49)
- Resuming from checkpoint: 1000/1934 rows (51.7%)
- 934 remaining * 12s = ~187 min (~3 hr) ETA ~14:50
- /no_think suffix active
- Log: results/log_qwen3_pheme.txt

### Current results (3/9 done)
| Dataset    | Model   | F1 Macro | Accuracy | Status              |
|------------|---------|----------|----------|---------------------|
| manchester | qwen3   | 0.5339   | 0.5616   | done                |
| monkeypox  | qwen3   | 0.5416   | 0.5470   | done                |
| pheme      | qwen3   | --       | --       | running 1000/1934   |
| manchester | llama33 | 0.7649   | 0.8384   | done                |
| monkeypox  | llama33 | --       | --       | BLOCKED (100K TPD)  |
| pheme      | llama33 | --       | --       | BLOCKED (100K TPD)  |

---
## [2026-05-25 19:51] -- llama33 TPD exhausted, awaiting midnight UTC reset

### Diagnosis: llama33 daily token cap hit
- After killing qwen3, llama33 immediately hit NEW 957s rate-limit waits (solo)
- This confirms it is NOT org-level interference -- it is llama33's own daily cap
- Cause: rogue PID 2264 ran ~22 hrs overnight, burned most of daily quota;
  today's Manchester run (365 tweets) used the remainder
- Pattern: ~940s waits = Groq's fixed "daily cap exhausted" retry signal
- Checkpoint safe: monkeypox_llama33 at 100/457 rows (100% JSON, 0 nulls)

### Plan: wait for Groq daily reset
- Groq resets at midnight UTC (~03:00 local time if UTC+3)
- Monitoring hourly; restart both llama33 + qwen3 PHEME after reset (sequentially)

---
## [2026-05-25 19:30] -- Parallel run causing interference, reverted to sequential

### Problem: org-level TPM exhausted by parallel run
- llama33: 110/457 Monkeypox, avg 769s/tweet (940s rate-limit waits) -- stuck
- qwen3 PHEME: 1030/1934, avg 63s/tweet -- partially OK but interfering
- Combined org consumption too high even with correct per-model req_delays

### Action taken
- Killed qwen3 PHEME (PID 10528) -- checkpoint saved at 1000/1934 rows
- llama33 (PID 60660) running SOLO -- next request (after current 967s wait) gets full TPM budget
- Plan: llama33 finishes all 3 datasets (~4.5 hr), then qwen3 PHEME resumes from 1000 rows (~3 hr)

### qwen3 PHEME progress saved
- 1000/1934 rows done (51.7%) -- 934 rows remaining
- /no_think suffix active -- at ~12s solo (est.) = ~187 min after llama33 finishes

---
## [2026-05-25 18:42] -- Daily reset confirmed, both models restarted

### Groq daily quota reset
- All 3 models tested: [OK] (gpt_oss 0.9s, llama33 0.6s, qwen3 1.4s)

### Running simultaneously (staggered 10s start)
- llama33 PID 60660 | Log: results/log_llama33.txt
  - Manchester: skipped (done)
  - Monkeypox: resuming from 50/457 rows
  - PHEME: pending after monkeypox + 90s cooldown
  - ETA: (407+1934) * 7s + 90s cooldown = ~273 min (~4.5 hr)
- qwen3 PHEME PID 10528 | Log: results/log_qwen3_pheme.txt
  - Resuming from 300/1934 rows
  - /no_think suffix active (cuts 10500->~600 tok/req)
  - ETA: 1634 * 12s = ~327 min (~5.5 hr)

### Safe to run in parallel because:
  - llama33 uses llama-3.3-70b-versatile TPM pool
  - qwen3 uses qwen/qwen3-32b TPM pool
  - Combined org-level: ~3300 + 2000 = 5300 tok/min (manageable)
  - Previously interference was from qwen3 running at 40 req/min (bug); now 5 req/min

---
## [2026-05-25 16:20] -- llama33 KILLED (daily TPD exhausted), awaiting reset

### Diagnosis: Groq daily TPD cap hit on llama33
- Monkeypox stuck at 50/457 -- rate limits escalating: 121s -> 351s -> 553s -> 980s -> 951s
- These are attempt 1/6 waits on NEW tweets -- not bursts, this is hard daily cap
- Root cause: original llama33 PID 2264 ran all night (5/24 12pm -> 5/25 9am) burning
  most of today's daily quota. Manchester (365 tweets) used the remainder.
- At 980s/tweet * 407 remaining = ~4.6 days wait. Killed PID 13776.
- Checkpoint intact: 50/457 monkeypox rows saved (100% JSON, 0 nulls)

### Plan: restart after Groq daily reset
- Groq resets at midnight UTC = ~03:00 local time
- Restart command (from C:\Users\yoni1\Desktop\ZEROSHO_CODE):
    =1
    Start-Process -FilePath python -ArgumentList @("scripts/run_all_cloud.py", "--model", "llama33") -RedirectStandardOutput "results/log_llama33.txt" -RedirectStandardError "results/log_llama33_err.txt" -NoNewWindow
- Will resume: monkeypox from 50 rows, then pheme from scratch
- ETA after reset: (407 + 1934) * 7s = ~272 min (~4.5 hr) + 90s cooldown between datasets

### What we have so far (3/9 complete)
| Dataset    | Model   | F1 Macro | Accuracy | Status    |
|------------|---------|----------|----------|-----------|
| manchester | qwen3   | 0.5339   | 0.5616   | done      |
| monkeypox  | qwen3   | 0.5416   | 0.5470   | done      |
| pheme      | qwen3   | --       | --       | SKIPPED   |
| manchester | llama33 | 0.7649   | 0.8384   | DONE      |
| monkeypox  | llama33 | --       | --       | paused 50/457 |
| pheme      | llama33 | --       | --       | pending   |

---
## [2026-05-25 15:58] -- manchester+llama33 DONE (F1=0.7649), Monkeypox running

### New result
| Dataset    | Model   | F1 Macro | Accuracy | Nulls | JSON |
|------------|---------|----------|----------|-------|------|
| manchester | llama33 | **0.7649** | **0.8384** | 0 | 365/365 |

- Huge improvement over qwen3 manchester (F1=0.5339 -> 0.7649, +0.231)

### Current status
- monkeypox + llama33: 50/457 (10.9%), avg 7.0s -- rate limits hit at transition
  - Pattern: Manchester->Monkeypox burst filled TPM window, attempt 2/6 waiting 553s
  - Self-recovering -- no action needed
- pheme + llama33: pending after monkeypox (~226 min at 7s/tweet)

### Updated results table
| Dataset    | Model   | F1 Macro | Accuracy | Status    |
|------------|---------|----------|----------|-----------|
| manchester | qwen3   | 0.5339   | 0.5616   | done      |
| monkeypox  | qwen3   | 0.5416   | 0.5470   | done      |
| pheme      | qwen3   | --       | --       | SKIPPED   |
| manchester | llama33 | 0.7649   | 0.8384   | DONE      |
| monkeypox  | llama33 | --       | --       | running   |
| pheme      | llama33 | --       | --       | pending   |

---
## [2026-05-25 15:21] -- qwen3 PHEME ABANDONED, llama33 STARTED

### qwen3 PHEME: abandoned (Groq daily TPD exhausted)
- /no_think fix: FAILED -- zero progress in 270s, constant rate limits unchanged
- Diagnosis: Groq daily token budget (TPD) for qwen/qwen3-32b is exhausted
  - Both req_delay=55s and /no_think prompt suffix had zero effect
  - This is a hard daily cap reset (not per-minute TPM), not fixable until midnight
- Decision: accept qwen3 2/3 results (manchester + monkeypox); PHEME skipped
  - Retry qwen3 PHEME after daily reset if time permits
  - Thesis note: qwen3 PHEME not available due to Groq free-tier daily TPD exhaustion
- Killed PID 31272

### llama33: STARTED (PID 13776)
- Log: results/log_llama33.txt
- Resumed Manchester checkpoint: 201/365 done (201 clean JSON rows)
- Running all 3 datasets: manchester -> monkeypox -> pheme
- ETA: ~164 + 457 + 1934 = 2555 remaining * 7s = ~298 min (~5 hr)
  - Manchester finish: ~19 min
  - Monkeypox finish: ~72 min
  - PHEME finish: ~298 min (~15:21 + 5hr = ~20:21)

### Current results table
| Dataset    | Model   | F1 Macro | Accuracy | Nulls | Status     |
|------------|---------|----------|----------|-------|------------|
| manchester | qwen3   | 0.5339   | 0.5616   | 0     | done       |
| monkeypox  | qwen3   | 0.5416   | 0.5470   | 0     | done       |
| pheme      | qwen3   | --       | --       | --    | SKIPPED    |
| manchester | llama33 | --       | --       | --    | running    |
| monkeypox  | llama33 | --       | --       | --    | pending    |
| pheme      | llama33 | --       | --       | --    | pending    |

---
## [2026-05-25 15:00] -- /no_think fix applied, llama33 checkpoint cleaned

### qwen3 PHEME: /no_think suffix added
- Root cause confirmed: PHEME tweets generate ~10,500 tokens/req (thinking blocks)
  - Single request exceeds entire 6000 TPM budget -> always rate-limited
  - req_delay is irrelevant when 1 request > TPM limit
- Fix: added prompt_suffix='/no_think' to qwen3 MODEL_CONFIG
  - Disables extended thinking -> cuts ~10,500 -> ~600 tok/req
  - req_delay reverted to 12s (correct for 600 tok/req)
- Killed PID 34880 (req_delay=55s, avg 162s/tweet, ETA 73hr)
- Restarted PID 31272 | Log: results/log_qwen3_pheme.txt
- Resumed from checkpoint: 300/1934
- Expected ETA if clean: 1634 * 12s = ~327 min (~5.5 hr)

### llama33 Manchester checkpoint cleaned
- Killed rogue PID 2264 (original parallel run, running since 5/24 12:03 PM)
- Was at 350 rows: 201 json (good) + 149 empty (rate-limit failures from 968s storm)
- Stripped checkpoint back to 201 clean JSON rows
- llama33 ready to start after qwen3 PHEME completes

---
## [2026-05-25 09:30] -- qwen3 PHEME restarted with req_delay=55s

### Fix applied
- Killed old process (PID 40108, req_delay=12s, avg 150s/tweet, ETA 71hr)
- Patched scripts/run_all_cloud.py: qwen3 req_delay 12 -> 55s
  - Root cause: <think> blocks ~3500 tokens + prompt/response ~1500 = ~5000 tok/req
  - 6000 TPM / 5000 tok = 1.2 req/min -> 55s between requests
- Restarted: python run_all_cloud.py --model qwen3 --dataset pheme
- PID 34880 | Log: results/log_qwen3_pheme.txt
- Resumed from checkpoint: 200/1934 rows saved
- New ETA: 1734 remaining * 55s = ~26 hours (finish ~11:30 tomorrow)

### Status
- 2/9 complete (manchester+monkeypox qwen3 done)
- pheme + qwen3: running, 200/1934 (10.4%), PID 34880

---
## [2026-05-25 00:46] -- 2/9 complete -- qwen3 PHEME: CRITICAL rate-limit issue

### Root cause: qwen3 thinking tokens much larger than estimated
- PHEME checkpoint: 220/1934 done (11.4%)
- Script reports: avg 150.3s/tweet | ETA ~4,293 min (~71 hours) -- IMPRACTICAL
- Rate limit pattern: ~190s waits on virtually every tweet
- Back-calculation: 190s retry_after -> ~25,000 tokens/60s window -> ~5,000 tokens/request
- Our req_delay=12s assumes ~1,067 tokens/req -- actually 5x too low
- Correct req_delay needed: 6000 TPM / 5000 tokens = 1.2 req/min -> need ~55s between requests
- Stored reasoning field is only ~375 chars (~94 tokens) -- the unlogged <think> block is ~3,500 tokens
- Rogue manchester_qwen process: GONE (completed or killed) -- not the cause

### Options
1. Kill qwen3 PHEME, restart with req_delay=55s -> ETA ~26 hrs (1714 * 55s)
2. Kill qwen3 PHEME, start llama33 now, revisit qwen3 PHEME later
3. Let it run at 150s/tweet -- ETA ~3 days (not recommended)

### Decided: monitoring -- user to decide on kill/restart

---
## [2026-05-24 21:12] -- 2/9 complete -- qwen3 PHEME starting

### Completed so far
| Dataset    | Model | F1 Macro | Accuracy | Nulls | JSON parses |
|------------|-------|----------|----------|-------|-------------|
| manchester | qwen3 | 0.5339   | 0.5616   | 0     | 362         |
| monkeypox  | qwen3 | 0.5416   | 0.5470   | 0     | 415         |

### Status
- qwen3 now running PHEME (1,934 samples)
- ETA: ~387 min at 12s/tweet (no rate limits) -- expect finish ~03:30
- Rate limits clearing from monkeypox tail burst
- llama33 starts after all 3 qwen3 summaries done

---
## [2026-05-24 21:03] — 1/9 complete — qwen3 rate-limit delay on Monkeypox tail

### Status
- ✅ **manchester + qwen3**: F1=0.5339 | Acc=0.5616 | nulls=0
- 🔄 **monkeypox + qwen3**: ~400+/457 rows done — hitting TPM rate limits on final tweets
  - Waits: 68s–188s per attempt (reading actual Groq retry_after values)
  - ETA: +30–90 min due to rate-limit pauses
  - Cause: some tweets generate long <think> blocks, bursting 6,000 TPM/min window
  - Rogue process manchester_qwen (notebook run with old model ID) is NOT causing this — it fails silently with empty responses
- ⏳ **pheme + qwen3**: starts after monkeypox; ~387 min at 12s/tweet (if no rate limits)
- ⏳ **llama33**: starts after all qwen3 done

---
## [2026-05-24 18:50] â€” 1/9 complete â€” qwen3 running solo (Monkeypox 250/457)

### Status
- âœ… **manchester + qwen3**: F1=0.5339 | Acc=0.5616 | nulls=0 | json_parses=362
- ðŸ”„ **monkeypox + qwen3**: 250/457 (54.7%) â€” no rate limit events, ~41 min remaining
- â³ **pheme + qwen3**: ~387 min after monkeypox completes
- â³ **llama33** (all 3): checkpoint at 200/365 Manchester; starts after all qwen3 done
- â“ **gpt_oss**: TPD exhausted; retry tomorrow or substitute model
- ðŸ—‘ï¸ Deleted stale `manchester_qwen_checkpoint.csv` (50 empty rows, old test artifact)

| Dataset | Model | F1 Macro | Accuracy | Nulls |
|---------|-------|----------|----------|-------|
| manchester | qwen3 | 0.5339 | 0.5616 | 0 |

---

## [2026-05-24 17:30] â€” 0/9 complete â€” qwen3 SOLO, others paused

### Root cause of earlier failures:
Running llama33 + qwen3 in **parallel** caused org-level TPM interference on Groq.
llama33 rate-limit waits grew: 95s â†’ 368s â†’ 570s â†’ 968s (classic rolling-window exhaustion).
gpt_oss TPD (200,000 tokens/day) was exhausted in the first aborted run (199,916/200,000 used).

### Strategy: Sequential runs only
1. âœ… **qwen3 solo** â€” running now (230/365 Manchester, ~5 hr total remaining)
2. â³ **llama33** â€” paused; checkpoint saved at 200/365 Manchester (100% JSON quality)
   - Start after qwen3 finishes (~22:30)
   - ETA: ~1.6 hr (resumes from checkpoint)
3. â“ **gpt_oss** â€” TPD exhausted; retry tomorrow OR replace with alternative model

### qwen3 timeline (solo, 7.1s/tweet):
| Dataset | Samples | Est. time | Start | End |
|---------|---------|-----------|-------|-----|
| Manchester | 365 (135 left) | ~16 min | 17:14 | ~17:46 |
| Monkeypox  | 457             | ~54 min  | ~17:46 | ~18:40 |
| PHEME      | 1,934           | ~229 min | ~18:40 | ~22:29 |

### llama33 timeline (sequential after qwen3, 2.3s/tweet):
| Dataset | Samples | Est. time | Start | End |
|---------|---------|-----------|-------|-----|
| Manchester | 165 left | ~6 min  | ~22:30 | ~22:36 |
| Monkeypox  | 457      | ~18 min | ~22:36 | ~22:54 |
| PHEME      | 1,934    | ~74 min | ~22:54 | ~00:08 |

### Checkpoints saved:
| Model | Dataset | Rows saved | Parse quality |
|-------|---------|-----------|---------------|
| llama33 | Manchester | 200/365 | 100% JSON, 0 nulls |
| qwen3   | Manchester | ~200/365 | In progress |

| Dataset | Model | F1 Macro | Accuracy | Null Predictions |
|---------|-------|----------|----------|-----------------|
| â€”       | â€”     | â€”        | â€”        | â€”               |



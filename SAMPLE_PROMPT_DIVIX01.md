You are reviewing a cryptocurrency market-microstructure mechanism for prospective confirmation. All evidence is synthetic. Use only the supplied facts; distinguish calculations, assumptions, hypotheses, and conclusions. Do not invent missing data.

## Claim and proposed strategy

“Aggressive buying depletes ask liquidity. Slow market-maker replenishment causes persistent upward pressure over the next 30 seconds, identifying a tradable causal mechanism.”

The strategy enters long after a qualifying burst and exits 30 seconds later.

## Event construction

Data cover 180 consecutive days, 12 assets, and three exchanges.

- An event starts at t when aggressive buy volume over the preceding two seconds exceeds its rolling 99th percentile, calculated from the preceding seven days.
- Events less than five seconds apart are merged. Thirty-second outcomes can overlap; shared market movements can trigger events across assets and exchanges.
- Ask depletion is measured from t−100 ms to t+100 ms.
- “Slow replenishment” means displayed ask depth never recovers to 80% of its pre-event level during [t, t+2 seconds]. Events without complete book coverage throughout that interval are excluded.
- Outcomes are mid-price returns from t to t+30 seconds. Reported net returns subtract a fixed 3 bp round-trip cost; execution prices were not simulated.
- Alignment uses exchange timestamps. Feed delays range from 20–400 ms and increase during high volatility.

## Selected-rule results

The system searched 2,400 threshold, horizon, liquidity-filter, and regime combinations using all 180 days.

| Metric | Result |
|---|---:|
| Qualifying events | 18,000 |
| Mean gross return | +5.8 bp |
| Mean assumed net return | +2.8 bp |
| Event-return standard deviation | 42 bp |
| IID event-bootstrap 95% interval for mean net return | [+2.2, +3.4] bp |
| Bootstrap resampling whole UTC days jointly across assets/exchanges | [−0.7, +6.3] bp |
| Share of total net P&L from five best days | 61% |
| Share from one exchange | 72% |

A retrospective split shows +3.7 bp net over the first 120 days and +1.0 bp over the final 60. Both the rule and regime definitions were selected using all 180 days. No prospective holdout exists.

## Regime evidence

Volatility uses the preceding five minutes. Returns below are gross over 30 seconds.

| Regime | Slow replenishment | Fast replenishment |
|---|---:|---:|
| High volatility | +8.0 bp; 12,000 events | +9.0 bp; 2,000 events |
| Low volatility | +1.4 bp; 6,000 events | +2.0 bp; 18,000 events |

Additional observations:

- Slow events have larger buy bursts and wider spreads.
- Incomplete-book exclusions disproportionately occur during volatility spikes.
- A news classifier labels 40% of slow events and 10% of fast events news-associated, using articles published up to ten minutes after t.
- A regression adjusting for burst size, volatility, spread, asset, and exchange estimates a +0.9 bp slow-replenishment coefficient, with day-clustered standard error 0.8 bp.
- Adding the subsequent two-second price return changes that coefficient to −0.4 bp.

## Assignment

Write a decision memo of at most 1,500 words:

1. **Calculate:** pooled gross means for slow and fast events, their difference, and both within-regime differences. Explain the discrepancy between pooled and conditional comparisons and its implications for the claim.

2. **Define estimands:** distinguish predictive, executable trading, and causal replenishment estimands. State when required signals become observable and which quantities the summaries identify.

3. **Analyze causality:** give a graph using named variables and directed arrows, including latent information shocks, aggressive flow, volatility, replenishment, price changes, and data availability. Explain how controlling for the subsequent two-second return could help or harm identification under different causal structures.

4. **Audit inference:** address outcome overlap, shared shocks, adaptive selection, concentration, missingness, timestamp uncertainty, and the retrospective split. Explain the limits of both confidence intervals. Do not invent an effective sample size.

5. **Challenge the explanation:** propose three competing explanations, each with a discriminating test, predicted result, and potential falsifier.

6. **Design confirmation:** propose the smallest credible prospective protocol with observable signal timing, executable prices and costs, a frozen primary hypothesis, evaluation unit, dependence-aware uncertainty, repeated-testing treatment, stopping rules, and pass/fail/inconclusive criteria. Explain sample-size planning without assuming event count alone determines information.

7. **Decide:** reject, revise and retest, or proceed to prospective confirmation. Identify the most consequential flaw and the next experiment with highest information value.

Prioritize defensible inference and falsifiability. Profitable association, causality, and executable trading performance are distinct claims.

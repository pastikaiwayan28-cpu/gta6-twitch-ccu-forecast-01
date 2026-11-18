# GTA VI Twitch CCU forecast (gta6-twitch-ccu-forecast-01)

Forecasts, hour by hour, how many people watch Grand Theft Auto VI live on
Twitch over the 24 UTC hours of its release day (2026-11-19), from everything
recorded up to the end of hour 1, as five quantiles per hour (0.05 / 0.25 /
0.5 / 0.75 / 0.95).

## Program

`main.py` (Python 3.13, standard library only). Run once per case; reads the
case JSON at `/input/<id>.in`, writes `/output/<id>.out`. Deterministic,
~0.2 s per case.

## Model

Conditional quantiles of the log-ratio to hour 1, learned from the case's
comparables (past big launches). For each hour h = 2..24 and each level tau,
the forecast is `p1 * exp(Q_tau(log(viewers_h / viewers_1)))`, where Q_tau is
a kernel-weighted empirical quantile over the comparables and p1 is hour 1's
print. The kernel weights comparables by similarity in log(hour-1 viewers)
(Gaussian, bandwidth 1.0); when the case is a rolling-midnight launch, rolling
comparables are upweighted (x5) and simultaneous ones downweighted (x0.2).

The shape is driven by when each region unlocks. Hour 1 (00:00-01:00 UTC) is
before most of the world unlocks for a rolling-midnight launch, so later hours
grow; for a simultaneous global unlock hour 1 is the peak and later hours
decay. The size of hour 1 is the strongest predictor of which regime a launch
is in.

## Validation

Leave-one-out over the 37 complete comparable launch curves in the kit
(each scored against its own true hours 2..24): mean pinball loss (log
scale) 0.405, versus 0.930 for the house persistence baseline on the same
set; on large-hour-1 launches (the regime of GTA VI and of the practice
cases) 0.199. The house baseline scores about 0.387 on the practice cases.

The practice cases' own ground truth is not in the kit (the server scores
them), so the practice loss is not computed locally; the comparables
leave-one-out above is the validation.

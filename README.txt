MARKET INTELLIGENCE TERMINAL v10.1
UNIFORM CLASSIFICATION + DELIVERY ACCELERATION + FPI SECTOR MONEY PUSH

WHAT IS NEW
- New FPI Sector Money Push tab based on official fortnightly sector-wise FPI data published by CDSL/NSDL.
- Refreshes only when you press Refresh from CDSL; normal dashboard startup stays fast.
- Saves the downloaded history locally and reuses it on later launches.
- Continuous sector graphs:
  1) cumulative FPI net equity flow (₹ crore)
  2) rolling 3-fortnight FPI net flow
  3) normalized FPI flow intensity (% of prior FPI AUC)
- FPI Money Push Score combines latest normalized flow, 3-fortnight intensity, buying consistency and flow acceleration.
- Latest fortnight inflow/outflow sector bar chart.
- Unified dashboard-sector view as well as original depository sector view.
- Overview shows a compact FPI sector money-push snapshot once history is available.
- Manual CSV import fallback when the depository site temporarily blocks automated access.

EXISTING v10 FEATURES RETAINED
- One uniform Stock -> Industry -> Sector master map used by all tabs.
- Delivery acceleration at stock and sector level.
- Broad NSE scanner, delivery, volume/RVOL, RS, accumulation and entry suitability.
- Selectable Piotroski threshold.
- TradingView links.
- Sector delivery/volume, sector RS, industry gain/loss, stock drill-down and news support.

HOW TO START ON WINDOWS
1. Extract this ZIP.
2. Double-click START_DASHBOARD.bat.
3. The first run may install Python packages.
4. The dashboard opens at http://localhost:8501

FIRST USE OF FPI TAB
1. Open FPI Sector Money Push.
2. Choose 24 or 36 fortnights.
3. Click Refresh from CDSL.
4. The data is saved locally. Future dashboard launches do not need to download the same history again.

INTERPRETATION
- Positive Net Flow = FPI net equity buying during that fortnight.
- Negative Net Flow = FPI net equity selling.
- Cumulative rising line = sustained FPI money push into that sector.
- Flow Intensity normalizes flow by prior FPI AUC, allowing smaller sectors to be compared with Financial Services and other large sectors.
- Use FPI flow as a slower institutional confirmation layer with Sector RS + Delivery + Volume, not as a standalone buy signal.

DATA NOTE
The official sector-wise FPI series is fortnightly. It is different from the daily aggregate FII/DII market activity number. Sector-wise FPI data is disseminated by the depositories (NSDL/CDSL).

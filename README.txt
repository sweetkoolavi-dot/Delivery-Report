MARKET INTELLIGENCE TERMINAL v9 — USABILITY + CLASSIFICATION UPGRADE

WHAT IS NEW
- Piotroski F-Score is now user-selectable. Choose the minimum score from 0 to 9 before the final quality gate.
- Important columns are frozen in key tables so they remain visible while scrolling horizontally:
  * Sector Stocks: Symbol, Signal, Accumulation Score
  * Sector Delivery + Volume: Industry, Sector Opportunity Score
  * Stock Opportunity Radar: Symbol, Signal, Entry Suitability, Accumulation Score
  * Final Accumulation table: Entry View, Opportunity Type, Symbol, Entry Suitability, Piotroski, Accumulation
- Sector Delivery + Volume ranking now includes Delivery Z Score.
- Sector Radar is permanently visible on the Overview page and uses a green gradient: darker green means higher opportunity score.
- Sector/industry mapping coverage is improved with a layered approach:
  1. Official Nifty Indices classification files where available
  2. Yahoo Finance structured profile lookup
  3. Moneycontrol fallback for unresolved high-priority names
  Lookup results are cached to protect speed.
- Broader sector/industry options are available in Sector Stocks after enrichment.
- TradingView links remain available throughout the stock views.
- High delivery is still not allowed to dominate ranking without volume, traded value and relative-strength confirmation.

HOW TO RUN ON WINDOWS
1. Extract the ZIP.
2. Open the folder.
3. Double-click START_DASHBOARD.bat
4. The launcher installs any missing packages and opens the dashboard at http://localhost:8501

RECOMMENDED WORKFLOW
1. Overview: market state, Opportunity Funnel, always-visible Sector Radar, rotation and Stock Opportunity Radar.
2. Industry Gain / Loss: see current-session industry strength and participation.
3. Sector Delivery + Volume: use Delivery Excess or Delivery Z Score plus volume and RS.
4. Sector Relative Strength: identify groups strengthening vs NIFTY 500.
5. Sector Stocks: drill into a sector/industry; important columns stay pinned while you scroll.
6. Accumulation Stocks: set delivery/volume/RS filters, choose your Piotroski threshold, then run the final quality gate.
7. TradingView + News: validate chart structure and check whether recent developments support the signal.

CLASSIFICATION / SPEED NOTE
The broad NSE scan is intentionally kept fast. The terminal does not make thousands of slow profile requests at startup. Official index mappings are loaded first; then the most relevant unmapped stocks are auto-enriched using Yahoo Finance and a Moneycontrol fallback. You can change the enrichment depth under Classification quality in the sidebar.

IMPORTANT
This is a research and shortlisting tool, not an automatic buy signal. High delivery percentage alone can be misleading when trading activity is small, so delivery abnormality is combined with volume, traded value, persistence, relative strength and entry-location logic.

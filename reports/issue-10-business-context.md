# Issue #10: Business Context and Value Proposition Report (Poland-First)

## Executive Objective
- Build a business-ready operating model to classify first-click session intent (`high`,`low`) and trigger actions that improve conversion with guardrails.
- Complement the data science work with market context, stakeholder accountability, process ownership, and KPI logic.
- Use Poland as the primary market context, Czechia/Lithuania/Slovakia/Romania as similar-market comparators, and the U.S. only as a benchmark reference.

## One-Slide Decision Summary
- Source-data reality:
  - UCI clickstream source is strongly Poland-centric: `country=29 (Poland)` accounts for `80.96%` of clicks and `81.50%` of sessions; `country=42 (USA)` is only `0.10%` of clicks and `0.11%` of sessions (derived from raw file in [S2]).
- Poland digital commerce demand:
  - In 2025, `69.7%` of people aged 16-74 in Poland purchased/ordered online in the last 12 months. [S3]
  - In 2025, online sales represented `9.1%` of total retail sales in Poland, with `25.2%` for `textiles, clothing, footwear`. [S4]
- Market pressure and opportunity:
  - Poland registered `252k` live births in 2024 (post-war low), which increases pressure on conversion efficiency in maternity retail demand capture. [S5]
- Similar-market signal:
  - Czechia and Slovakia had all regions above the EU average in online shopping participation (2023 regional view), while Lithuania and Romania were below average in all regions. [S8]
  - Lithuania still showed strong long-run adoption growth in internet-user online purchasing (`36%` to `72%`, 2014-2024). [S7]
- U.S. reference only:
  - U.S. e-commerce share was `16.4%` of retail sales in Q3 2025 (benchmark context, not operating base). [S13]

## Business Context (Poland Core + Similar Markets)
### Poland Core Context
- Consumer readiness:
  - `69.7%` online purchase incidence (16-74), `96.2%` household internet access (2025). [S3]
- Channel economics:
  - Online share of all retail at `9.1%` (2025), with higher online intensity in apparel-related category (`25.2%` in textiles/clothing/footwear). [S4]
- Demand-side demographic backdrop:
  - `252k` births in 2024; birth rate `6.7‰`; natural increase `-157k`. [S5]

### Similar Markets (Priority Comparator Set)
- Czechia:
  - Market turnover reached `CZK 194 billion` in 2024 (`+5%` YoY) from Heureka/APEK industry tracking. [S10]
  - Trade.gov framing also positions Czech e-commerce as a high-adoption market (older but useful baseline). [S11]
- Lithuania:
  - Trade.gov describes a robust, mobile-led, high-penetration e-commerce market with active domestic and cross-border dynamics. [S12]
- Slovakia and Romania:
  - Regional Eurostat pattern shows divergent maturity but meaningful growth runway in parts of CEE. [S8]

### U.S. Benchmark Reference
- Use U.S. metrics for comparison of maturity and operational ambitions only, not for direct transfer of behavior assumptions. [S13]

## Market Sizing Emphasis (Poland-First)
| Sizing Layer | Definition | External Anchor | Business Use |
|---|---|---|---|
| Core SOM (Poland) | Near-term value from intent-led interventions in Poland traffic | Poland online retail share `9.1%`; apparel-related online share `25.2%` [S4] | 12-month operating plan and pilot economics |
| Core SAM (Poland digital shoppers) | Addressable online maternity shoppers in Poland | `69.7%` online buyer penetration (16-74) [S3] | Audience sizing and budget allocation |
| Adjacent SAM+ (CEE comparators) | Expansion-ready neighboring/analog markets | Czech turnover `CZK 194bn` (2024) [S10]; Lithuania growth profile [S12]; Eurostat regional patterns [S8] | Sequenced market expansion logic |
| Reference TAM signal | International maturity benchmark | U.S. e-commerce share `16.4%` (Q3 2025) [S13] | Board-level benchmark, not immediate plan |

### Practical Sizing Formulas (Placeholders)
- `poland_incremental_revenue = eligible_sessions_PL x baseline_CVR x uplift x AOV`
  This estimates extra revenue in Poland from the intent strategy.
  - eligible_sessions_PL: Poland sessions where your intervention can be applied
  - baseline_CVR: normal conversion rate without intervention
  - uplift: expected conversion improvement from intervention (as a decimal, e.g. 0.08 for +8%)
  - AOV: average order value
- `poland_incremental_margin = poland_incremental_revenue x gross_margin_rate - incentive_cost - media_cost`
  This estimates extra profit (margin), not just sales.
  - Start with incremental revenue
  - Keep only gross margin portion (gross_margin_rate)
  - Subtract promo/discount costs (incentive_cost)
  - Subtract paid traffic/activation costs (media_cost)
- `cee_expansion_value = sum(country_i_eligible_sessions x country_i_uplift x country_i_AOV x country_i_margin)`
  This estimates total expansion value across CEE markets by adding each country's contribution.
  - country_i_eligible_sessions: sessions in country i where the intervention can run
  - country_i_uplift: expected conversion improvement in country i
  - country_i_AOV: average order value in country i
  - country_i_margin: gross margin rate in country i
  - sum(...): add all country-level values into one regional total
  - Consistency note: define uplift once (incremental, e.g. 0.08, or multiplier, e.g. 1.08) and use the same convention in every formula.

## User Behavior and Decision Environment (Poland-Weighted)
- Consumer behavior (Poland market evidence):
  - Buyers are price-sensitive and comparison-driven; over `70%` use price comparison services. [S9]
  - Delivery convenience matters; parcel lockers and courier remain key preferences. [S9]
  - Most purchased categories include clothing/accessories (`79%`) and shoes (`66%`), supporting apparel relevance. [S9]
- Friction and abandonment baseline:
  - Global checkout abandonment remains high (`70.19%` average), with major drivers including extra costs and delivery concerns. [S14]
- Business implication:
  - Intervention design should prioritize price transparency, delivery clarity, and low-friction checkout experiences before aggressive incentives.

## Dataset Granularity Reality Check (UCI Variables Table)
- Data available:
  - `165,474` click records, `24,026` sessions, `14` variables, single e-shop, Apr-Aug 2008. [S1][S2]
  - Useful features include sequence (`order`), product/category (`page 1`, `page 2`), offer context (`price`, `price 2`), and placement (`location`, `model photography`).
- Data limits:
  - No explicit purchase/outcome variable in the variables table (weak supervision problem). [S2]
  - No click timestamp (sequence only), limiting dwell-time/inter-click timing features.
  - No persistent customer ID across sessions.
  - Country dimension includes non-country buckets (`biz/com/int/net/org`), reducing geo precision. [S2]
  - Strong Poland skew reduces direct transferability to U.S. behavior assumptions.
- Decision implication:
  - Treat phase 1 as a Poland-first intent-proxy program with experiment-led validation before CEE expansion and before any U.S.-style scaling assumptions.

## Stakeholder Map
| Stakeholder | Primary Goal | Decision Rights | Dependencies | KPI Ownership |
|---|---|---|---|---|
| Merchandising | Grow conversion and margin in apparel categories | Incentive depth, product visibility, assortment logic | DS score quality, stock, gross margin rules | Conversion by tier, margin delta |
| Growth/CRM | Efficient traffic monetization | Trigger policies by tier/channel | Audience sync, suppression rules, creative supply | CAC by tier, uplift |
| Product/Engineering | Fast and low-friction intent activation | UX placement and latency tradeoffs | Event quality, model service reliability | Funnel completion, latency |
| Data Science | Robust early-intent scoring with weak labels | Proxy definition, threshold policy, drift controls | Event quality, holdout design | Precision@K, proxy stability |
| Legal/Privacy | Compliant targeting in sensitive context | Consent/suppression standards, policy boundaries | Product implementation and vendor controls | Compliance incident rate |
| Ops/CX | Protect promise quality and trust | Delivery/returns message standards | Fulfillment reliability and policy consistency | Complaint rate, return delta |

## Business Process and Steps (Execution Checklist)
| Step | Owner | Trigger | Action | SLA | Failure Mode | KPI |
|---|---|---|---|---|---|---|
| 0. Proxy label and geography gate | DS + Business owners | Pre-model | Define proxy labels + Poland-first scope assumptions | 2-week design | Proxy weakly linked to outcomes | Proxy-to-order correlation |
| 1. Session signal capture | Product + DS | First events | Capture first-click features from available variables | `<200ms` ingestion | Missing events | Event completeness |
| 2. Real-time tiering | DS + Eng | First-click window reached | Score into `high/medium/low` | `<500ms` scoring | Timeout or stale model | Scoring success rate |
| 3. Tiered action orchestration | Growth + Merchandising | Tier emitted | Trigger policy-safe actions by intent tier | Same session | Wrong action per tier | Incremental uplift by tier |
| 4. Guardrail check | Legal/Privacy + Product | Pre-render | Enforce consent/suppression/risk rules | Hard block | Policy bypass | Guardrail breach count |
| 5. Outcome loop | DS + Business owners | Daily/weekly cadence | Evaluate uplift, margin, returns, CX and recalibrate | Weekly | Local uplift with net-profit loss | Net margin uplift |
| 6. Similar-market readiness gate | Leadership + DS + Growth | Post-pilot | Validate transferability to Czechia/Lithuania/Slovakia/Romania | Monthly review | Overfitting to Poland behavior | Transferability index |

## KPI Template (Placeholders Only)
- Model and data quality:
  - `precision_at_k = true_positives_at_k / predicted_positives_at_k`
  - `proxy_label_stability = consistent_proxy_labels / total_labeled_sessions`
  - `country_transferability_index = performance_comparator_market / performance_poland`
- Commercial:
  - `incremental_conversion_uplift = (conv_treatment - conv_control) / conv_control`
  - `gross_margin_impact = gross_margin_treatment - gross_margin_control`
- Risk/CX:
  - `return_rate_delta = return_rate_treatment - return_rate_control`
  - `cx_guardrail_index = weighted(opt_out_rate, complaint_rate, suppression_error_rate)`
- Placeholder targets:
  - `Target_Precision@K = [TBD]`
  - `Target_Conversion_Uplift = [TBD]`
  - `Minimum_Transferability_Index = [TBD]`
  - `Max_Complaint_Rate = [TBD ceiling]`

## Risks, Governance, and Guardrails
| Risk | Why It Matters | Control Design | Owner |
|---|---|---|---|
| Poland overfitting risk | Model policies may fail in adjacent markets | Explicit transferability gate before scale | DS + Leadership |
| Weak-label risk | No direct purchase label in source schema | Proxy governance + holdout testing | DS |
| Margin leakage | Incentives may target already-converting sessions | Uplift-based incentive policy, strict holdouts | Merchandising + Finance |
| UX fatigue | Over-targeting harms trust and retention | Frequency caps and suppression logic | Growth + Product |
| Regulatory misalignment | EU e-commerce/GDPR/DMA/DSA obligations affect design | Legal review in policy lifecycle | Legal/Privacy [S9][S17] |

## Inferred Acceptance Criteria
- Poland-first business context is explicit and data-backed.
- Similar-market framing (Czechia/Lithuania/Slovakia/Romania) is concrete, not generic.
- U.S. is used as benchmark only.
- Dataset granularity limits and geography skew are clearly operationalized into process gates.
- Market sizing section includes actionable TAM/SAM/SOM-like structure and formulas.

## Sources
- [S1] UCI dataset page: https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping
- [S2] UCI dataset archive and variables file: https://archive.ics.uci.edu/static/public/553/clickstream%2Bdata%2Bfor%2Bonline%2Bshopping.zip
- [S3] Statistics Poland, Information society in Poland in 2025 (69.7% online buyers): https://stat.gov.pl/en/topics/science-and-technology/information-society/information-society-in-poland-in-2025%2C2%2C15.html
- [S4] Statistics Poland, Internal market (2025 online retail share 9.1%; textiles/clothing/footwear 25.2%): https://ssgk.stat.gov.pl/Rynek_wewnetrzny.html
- [S5] Statistics Poland, Population and vital statistics in 2024 (252k births, birth rate 6.7‰, natural increase -157k): https://stat.gov.pl/files/gfx/portalinformacyjny/en/defaultaktualnosci/3286/3/37/1/population_size_and_structure_and_vital_statistics_in_poland_by_territorial_division_in_31-12-2024.pdf
- [S6] Statistics Poland, Demographic situation 2025 preliminary estimates: https://ssgk.stat.gov.pl/Ludnosc.html
- [S7] Eurostat, Online shopping in the EU keeps growing (2024): https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20250220-3
- [S8] Eurostat, Regional online shopping patterns (2023): https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20241129-2
- [S9] U.S. Commercial Service, Poland eCommerce guide (market size and behavior): https://www.trade.gov/country-commercial-guides/poland-ecommerce
- [S10] Heureka Group / APEK, Czech e-commerce turnover 2024: https://heureka.group/cz-en/about-us/group-news/press-releases/czech-e-commerce-ended-2024-with-a-total-turnover-of-czk-194-billion-up-5-year-on-year/
- [S11] U.S. Commercial Service, Czech Republic eCommerce guide: https://www.trade.gov/country-commercial-guides/czech-republic-ecommerce
- [S12] U.S. Commercial Service, Lithuania eCommerce guide: https://www.trade.gov/country-commercial-guides/lithuania-ecommerce
- [S13] U.S. Census, Quarterly retail e-commerce sales (Q3 2025): https://www.census.gov/retail/ecommerce.html
- [S14] Baymard Institute, Cart abandonment benchmark (2025 update): https://baymard.com/blog/ecommerce-checkout-usability-report-and-benchmark
- [S15] NIST AI RMF 1.0: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10
- [S16] FTC BetterHelp final order (sensitive data enforcement reference): https://www.ftc.gov/news-events/news/press-releases/2023/07/ftc-gives-final-approval-order-banning-betterhelp-sharing-sensitive-health-data-advertising
- [S17] European Commission, EU e-commerce rules / DSA-DMA context: https://digital-strategy.ec.europa.eu/en/policies/e-commerce-rules-eu

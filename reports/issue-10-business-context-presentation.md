# Issue #10 Executive Brief (Poland-First, CEE-Aware)

## 1-Page Executive Summary
- Strategic objective:
  - Predict early-session purchase intent and trigger actions that increase conversion and margin while preserving trust.
- Geographic truth from source data:
  - UCI session data is Poland-dominant: `81.50%` of sessions and `80.96%` of clicks are from `country=29 (Poland)`; U.S. is only `0.11%` of sessions (`0.10%` clicks), derived from [S2].
- Poland demand and channel context:
  - `69.7%` of people aged 16-74 bought online in Poland in 2025. [S3]
  - Online sales were `9.1%` of all retail in 2025; `25.2%` for textiles/clothing/footwear. [S4]
  - 2024 births were `252k` (down), increasing pressure on conversion efficiency and precision targeting in maternity commerce. [S5]
- Similar-market cluster for expansion:
  - Czechia: `CZK 194bn` e-commerce turnover in 2024 (`+5%` YoY). [S10]
  - Lithuania: strong digital-commerce momentum and high internet commerce participation growth. [S7][S12]
  - Slovakia/Romania: material variation in regional maturity, requiring country-specific rollout pacing. [S8]
- U.S. reference only:
  - U.S. e-commerce share `16.4%` in Q3 2025 is a benchmark for maturity, not a transfer assumption. [S13]

## Market Sizing Emphasis (Decision View)
| Layer | Scope | Anchor | Use |
|---|---|---|---|
| Core SOM | Poland intent-led value capture | Poland retail online intensity + apparel online intensity [S4] | 12-month pilot economics |
| Core SAM | Poland online maternity audience | `69.7%` online buyers (16-74) [S3] | Budget and targeting scope |
| Adjacent SAM+ | Czechia/Lithuania/Slovakia/Romania | Czech turnover + Eurostat regional maturity signals [S8][S10][S12] | Sequenced expansion plan |
| Reference benchmark | U.S. e-commerce maturity | `16.4%` retail e-commerce share [S13] | Ambition calibration |

- Formula templates:
  - `poland_incremental_revenue = eligible_sessions_PL x baseline_CVR x uplift x AOV`
  - `poland_incremental_margin = revenue x gross_margin_rate - incentive_cost - media_cost`

## Data Granularity Reality Check (Critical)
- Available:
  - Session sequence and product-level click context exist (`order`, category/model, price, placement). [S2]
- Missing:
  - No explicit purchase outcome variable, no click timestamp, no cross-session customer ID. [S2]
  - Country field includes non-country buckets (`biz/com/int/net/org`). [S2]
- Business implication:
  - Run a Poland-first weak-label pilot.
  - Require transferability validation before similar-market rollout.

## Stakeholder Ownership (Condensed)
| Stakeholder | Owns | Core KPI |
|---|---|---|
| Merchandising | Incentive depth + assortment logic by intent tier | Margin delta, conversion |
| Growth/CRM | Trigger orchestration and suppression rules | Uplift, CAC by tier |
| Product/Engineering | Real-time scoring UX delivery | Latency, funnel completion |
| Data Science | Proxy labels, thresholds, drift and transferability | Precision@K, transferability index |
| Legal/Privacy | Policy boundaries and consent controls | Compliance incidents |
| Ops/CX | Promise quality and complaint containment | Return delta, complaint rate |

## 90-Day Business Process
- Days 1-30:
  - Define Poland-first proxy labels, policy guardrails, and pilot KPIs.
- Days 31-60:
  - Run controlled A/B pilot in Poland with holdout and margin/CX constraints.
- Days 61-90:
  - Approve expansion only if uplift, margin, and transferability thresholds are met.

## KPI Template (Placeholders)
- `precision_at_k`
- `incremental_conversion_uplift`
- `gross_margin_impact`
- `country_transferability_index`
- `cx_guardrail_index`
- Targets:
  - `Target_Precision@K = [TBD]`
  - `Target_Conversion_Uplift = [TBD]`
  - `Minimum_Transferability_Index = [TBD]`
  - `Max_Complaint_Rate = [TBD]`

## Sources
- [S1] UCI dataset page: https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping
- [S2] UCI dataset archive and variables file: https://archive.ics.uci.edu/static/public/553/clickstream%2Bdata%2Bfor%2Bonline%2Bshopping.zip
- [S3] Statistics Poland, Information society in Poland in 2025: https://stat.gov.pl/en/topics/science-and-technology/information-society/information-society-in-poland-in-2025%2C2%2C15.html
- [S4] Statistics Poland, Internal market: https://ssgk.stat.gov.pl/Rynek_wewnetrzny.html
- [S5] Statistics Poland, Population and vital statistics in 2024: https://stat.gov.pl/files/gfx/portalinformacyjny/en/defaultaktualnosci/3286/3/37/1/population_size_and_structure_and_vital_statistics_in_poland_by_territorial_division_in_31-12-2024.pdf
- [S7] Eurostat, Online shopping in EU keeps growing: https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20250220-3
- [S8] Eurostat, Regional online shopping patterns: https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20241129-2
- [S10] Heureka/APEK, Czech e-commerce turnover 2024: https://heureka.group/cz-en/about-us/group-news/press-releases/czech-e-commerce-ended-2024-with-a-total-turnover-of-czk-194-billion-up-5-year-on-year/
- [S12] U.S. Commercial Service, Lithuania eCommerce: https://www.trade.gov/country-commercial-guides/lithuania-ecommerce
- [S13] U.S. Census e-commerce report: https://www.census.gov/retail/ecommerce.html

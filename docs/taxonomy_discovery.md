# Tesco Customer Support: Inductive Taxonomy Discovery (Draft)

> **Status**: **Draft Human Taxonomy** (Phase 2).  
> **Note**: This taxonomy was inductively derived from qualitative observation of the 200-conversation stratified sample (`outputs/samples/conversation_sample_200.md`) extracted in Phase 1. It is subject to empirical revision and refinement following independent double-annotation and formal disagreement adjudication.

---

## 1. Discovery Methodology

Rather than imposing a top-down e-commerce schema, the taxonomy was discovered through open coding of 200 real Tesco customer support Twitter threads spanning 2014–2017.

Observations from the data:
1. **Physical Supermarket vs Online Grocery**: Tesco operates both large physical superstores/express shops and an online grocery delivery/Click & Collect service. Customer complaints divide sharply between in-store incidents (till queues, colleague interactions, store facilities, shelf stock) and home delivery logistics (missing bags, delivery slot availability, driver delays).
2. **Food Freshness & Safety Urgency**: A substantial portion of interactions involves spoiled fresh produce, foreign objects, and packaging defects requiring rapid reimbursement and supplier barcode logging.
3. **Loyalty Program Complexity**: Tesco Clubcard is a major ecosystem component. Login errors, missing point statements, and voucher redemption issues form a distinct, high-frequency inquiry stream.
4. **Social & Banter Context**: Twitter customer support includes non-complaint chatter, store praise, and humorous commentary on marketing campaigns.

---

## 2. Draft Taxonomy: 8 Primary Intents

| Intent Label | Definition & Core Scope | Typical Keywords & Customer Phrases | Example Real Conversation IDs |
| :--- | :--- | :--- | :--- |
| `delivery_and_orders` | Online grocery delivery, Click & Collect orders, missing/substituted items, delivery driver conduct, delivery slot booking difficulties. | *"no delivery slots"*, *"order cancelled"*, *"Click & Collect bags"*, *"driver arrived late"*, *"missing item from delivery"* | `1019434`, `1560636`, `1612297`, `1629894`, `1770985` |
| `product_quality_and_safety` | Rotten/moldy food, foreign bodies (insects, plastic, glass), damaged packaging, short expiration dates, bad taste/contamination. | *"tastes like lemon fairy liquid"*, *"red bits in squash"*, *"short dates"*, *"extra free bit in donuts"*, *"empty can in multipack"* | `153811`, `1476540`, `1652107`, `1861903`, `2185119` |
| `stock_and_availability` | Inquiries regarding product availability, out-of-stock items, discontinued product lines, midnight entertainment/game launches. | *"stopped selling quorn nuggets?"*, *"no cat litter in Solihull"*, *"midnight launch for Call of Duty"*, *"where are the books?"* | `1172134`, `1458193`, `1608718`, `2323673` |
| `store_experience_and_staff` | Physical store conditions, staff compliments or complaints, checkout queues, store cafe/petrol station facilities, in-store hazards. | *"checkout colleague was an utter joy"*, *"waiting 10 minutes at shutter"*, *"cafe food served wrong"*, *"leek in aisle 3"* | `760248`, `1401451`, `1449888`, `1462450`, `1629094`, `1665568` |
| `pricing_promotions_and_vouchers` | Overcharging at till, shelf vs till price mismatch, multi-buy offer errors, paper voucher scanning failures, fuel price queries, price matching. | *"why is diesel more expensive in Stoke"*, *"scanned £5 voucher on till which crashed"*, *"£20 for 2 packs or £22 in Scotland?"* | `1385316`, `1478115`, `1481958` |
| `clubcard_and_loyalty` | **All Clubcard-related topics**: Clubcard account login, points balance, missing quarterly vouchers, Clubcard Plus, card replacement. | *"Clubcard issue I had"*, *"points missing"*, *"great service from Chris on clubcard team"*, *"can't log into Clubcard app"* | `108311`, `1269311` |
| `website_and_app_technical` | **Non-Clubcard technical failures**: Online grocery website crashes, general app bugs, checkout payment gateway failures, basket errors. | *"site unusable"*, *"won't give me an option to pay"*, *"iPad app greyed out"*, *"turned your shopping site into a mess"* | `1364913`, `1435199`, `1620460` |
| `general_feedback_and_chitchat` | Environmental/packaging feedback, lighthearted banter, compliments without specific store details, responses to Tesco promotional tweets, off-topic tweets. | *"why so much packaging on fish"*, *"Christmas cake baking day"*, *"typo on packaging"*, *"rearranged herbs to say DONGBAGS"* | `992960`, `1033307`, `1756608`, `2736807` |

---

## 3. Mutual Exclusivity Boundary: Clubcard vs Technical

> [!IMPORTANT]
> **Strict Separation Rule**:
> - Any issue concerning **Clubcard points, Clubcard account login, Clubcard app, Clubcard vouchers, or Clubcard promotions** MUST be categorized as `clubcard_and_loyalty`, regardless of whether a website or app was used.
> - `website_and_app_technical` is reserved **strictly for non-Clubcard digital issues** (e.g. general grocery shopping cart errors, payment gateway timeouts, general website UI redesign feedback).

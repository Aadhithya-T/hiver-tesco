# Tesco Customer Support Annotation Guidelines (Phase 2)

> **Document Version**: 1.0 (Draft)  
> **Applicability**: Golden-set candidate annotation (200 conversations)  
> **Status**: Draft Human Taxonomy subject to revision after independent annotation and formal adjudication.

---

## 1. Objective

The goal of this annotation task is to establish a rigorous, human-labelled golden dataset of 150–250 Tesco Twitter support conversations. Each conversation is annotated at the conversation level with:
1. **Primary Intent** (exactly one label from the draft taxonomy)
2. **Escalation Decision** (`escalate` vs `do_not_escalate`)
3. **Escalation Rationale** (reason for escalation, if applicable)
4. **Evidence Notes** (strictly redacted / generic factual entities)
5. **Ambiguity Notes** (optional flags for multi-intent or borderline cases)

---

## 2. Field Definitions & Permitted Values

### Field 1: `primary_intent` (Required)
Annotators must select **exactly one** primary intent that captures the core driving reason for the customer's contact.

Allowed labels (Draft Taxonomy):
- `delivery_and_orders`: Grocery delivery delays, slot availability, missing groceries, Click & Collect.
- `product_quality_and_safety`: Spoiled food, foreign objects, packaging defects, short dates.
- `stock_and_availability`: Store stock checks, discontinued lines, product launches.
- `store_experience_and_staff`: Staff conduct, checkout queues, store cleanliness, cafe/facility issues.
- `pricing_promotions_and_vouchers`: Overcharging, shelf vs till discrepancies, voucher scan failures.
- `clubcard_and_loyalty`: Points issues, missing reward vouchers, Clubcard app login, card replacement.
- `website_and_app_technical`: General non-Clubcard website/app bugs, payment checkout gateway errors.
- `general_feedback_and_chitchat`: Compliments, packaging feedback, social banter, marketing roots.

#### Mutual Exclusivity Boundary: `clubcard_and_loyalty` vs `website_and_app_technical`
- If an issue involves **Clubcard** (even if occurring on the mobile app or website, such as login or voucher display), it **MUST** be labeled `clubcard_and_loyalty`.
- `website_and_app_technical` is reserved **strictly for non-Clubcard systems** (e.g., grocery basket checkout, general website redesign, payment gateway crash).

---

### Field 2: `escalation_needed` (Required)
Determines whether the interaction requires private human agent intervention (escalation) or can be fully resolved/handled publicly.

Allowed values:
- `escalate`: The issue cannot be resolved publicly. It requires private communication (DM) to collect personal identifiable information (PII), access backend customer accounts, issue financial refunds/vouchers, or conduct an internal food safety or staff investigation.
- `do_not_escalate`: The inquiry is purely informational, public policy clarification, praise/compliment, casual banter, or general feedback that requires no account lookup or financial restitution.

---

### Field 3: `escalation_rationale` (Required if `escalation_needed == escalate`, else `none`)
Specifies the operational justification for escalating:

| Rationale Value | Trigger Criteria |
| :--- | :--- |
| `requires_pii_or_dm` | Agent must request customer name, postcode, order number, or account details via Direct Message. |
| `refund_or_compensation` | Customer is owed financial reimbursement, MoneyCard, or voucher credit. |
| `product_safety_investigation` | Foreign object in food, chemical contamination, severe food poisoning, or product recall. |
| `formal_complaint` | Severe colleague misconduct, discrimination, or formal escalation to store manager. |
| `technical_support` | Backend account lockout, server-side glitch, or payment transaction failure requiring IT support. |
| `none` | Used when `escalation_needed == do_not_escalate`. |

---

### Field 4: `evidence_notes` (Optional / Text)

> [!CAUTION]
> **STRICT PII PROHIBITION**:
> **NEVER copy customer names, telephone numbers, email addresses, residential postcodes, or full order numbers into this field.**  
> Annotators MUST use generic descriptors or redacted placeholders.

**Permitted formats**:
- `"[Postcode provided in thread]"`
- `"[Order number provided]"`
- `"Store: Hammersmith Superstore"` (public store names are permitted)
- `"Product: Romano Chicken Pizza"` (public product names are permitted)
- `"Barcode: [Barcode provided]"`
- `"Expiry: 2017-11-01"`

---

### Field 5: `ambiguity_notes` (Optional / Text)
Use this field if the conversation is borderline, touches multiple topics, or contains unusual anomalies (e.g., *"Customer started with delivery complaint but also complained about squash taste"*).

---

## 3. Multi-Intent Precedence Hierarchy

When a customer expresses multiple intents in a single thread, resolve the primary intent using the following precedence rules:

1. **Safety & Health Precedence**: If a thread mentions foreign objects, food poisoning, or safety hazards, classify as `product_quality_and_safety` regardless of other complaints.
2. **Direct Financial Loss Precedence**: Overcharging, missing orders, or lost vouchers take precedence over general feedback or stock queries.
3. **Delivery Logistics vs Product**: If a delivered item was damaged/missing, classify under `delivery_and_orders` if the issue arose during home delivery, or `product_quality_and_safety` if the manufactured product itself was defective.
4. **Core Grievance over Banter**: Sarcasm or opening jokes are secondary to the underlying operational complaint.

---

## 4. Special Cases

### Marketing & Broadcast Root Tweets
- Tweets where the conversation root was authored by Tesco (e.g. Christmas food advertisements, baking recipes):
  - If customers reply with playful comments, memes, or casual questions: classify as `general_feedback_and_chitchat` and `do_not_escalate`.
  - If a customer hijacks the marketing thread to complain about an actual unfulfilled delivery or spoiled item: classify according to the customer's actual operational intent and evaluate escalation accordingly.

### Competitor & Cross-Brand Mentions
- Tweets mentioning competitor supermarkets (`@sainsburys`, `@asda`, `@morrisons`):
  - If comparing prices or price match guarantee: `pricing_promotions_and_vouchers`.
  - If expressing general preference or jokes: `general_feedback_and_chitchat`.
  - If complaining about Tesco service while threatening to switch to a competitor: classify by the underlying Tesco service issue (e.g., `delivery_and_orders` or `store_experience_and_staff`).

---

## 5. Independent Annotation & Adjudication Protocol

To guarantee high scientific reliability:
1. **Independent Dual Annotation**: Two human annotators label the 200 candidate conversations independently using the provided CSV template without consulting each other.
2. **Inter-Annotator Agreement**: Compute Cohen's Kappa ($\kappa$) on `primary_intent` and `escalation_needed`. A target agreement of $\kappa \ge 0.75$ indicates a reliable taxonomy.
3. **Disagreement Adjudication**: Any conversation with discrepant labels is reviewed in a joint adjudication session with a lead adjudicator.
4. **Taxonomy Evolution**: If a pattern of disagreements reveals ambiguity, the draft taxonomy definitions or boundary rules will be updated before finalizing the golden benchmark for Phase 3.

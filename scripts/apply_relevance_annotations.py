"""Script to populate verified human relevance judgements on dev review sample."""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hiver_tesco.retrieval.relevance_validator import validate_relevance_file

annotations = {
    # Query 1444616: merged account cannot use vouchers
    1: ("irrelevant", "BM25 keyword overlap on filler words; candidate is about poor salad bags"),
    2: ("irrelevant", "BM25 match on 'use', 'handle'; candidate is about gaming monitor"),
    3: ("irrelevant", "BM25 match on 'we', 'cannot'; candidate is about Irish Twitter account DM issue"),
    4: ("partially_relevant", "Customer voucher not working on system; Tesco investigates voucher details"),
    5: ("partially_relevant", "Customer complaints about promised vouchers not received; Tesco assists via DM"),
    6: ("irrelevant", "Customer requesting to opt out of vouchers mailing; opposite of voucher redemption error"),

    # Query 1017391: waiting for sofa collection from Tesco Direct
    7: ("partially_relevant", "Customer waiting for delayed Tesco response/refund; general delayed service resolution"),
    8: ("partially_relevant", "Customer waiting for click & collect order sent back; delivery/collection failure"),
    9: ("irrelevant", "Stock availability issue for prawn cocktail crisps; unrelated to furniture collection"),
    10: ("relevant", "Tesco Direct sofa delivery and service disaster; Tesco requests order details to resolve"),
    11: ("partially_relevant", "Click and collect collection issue; general collection dispute"),
    12: ("irrelevant", "Positive feedback for rapid customer service phone callback; unrelated"),

    # Query 1532287: delivery saver Christmas delivery slot pre-booking dates
    13: ("relevant", "Exact query regarding Christmas grocery delivery saver slot release dates; Tesco explains late Nov timeframe"),
    14: ("partially_relevant", "Delivery saver availability complaint; Tesco offers account check and refund inquiry"),
    15: ("relevant", "Delivery saver eligibility and priority access for Christmas delivery slots; Tesco provides booking date"),
    16: ("relevant", "Exact query on delivery saver Xmas slots opening; Tesco provides exact cut-off (23 Nov) and booking dates (26 Nov)"),
    17: ("relevant", "Exact query on release dates for Xmas delivery saver slots; Tesco confirms timeline and email notice"),
    18: ("relevant", "Delivery saver Xmas slot release dates; Tesco requests account info to provide personalized key dates"),

    # Query 1365750: personal trainer joke asking to ban someone from store
    19: ("irrelevant", "Inquiry about toy sale dates; unrelated"),
    20: ("irrelevant", "Compliment about Aylesbury store manager; unrelated"),
    21: ("irrelevant", "Product availability check in Aberdeen; unrelated"),
    22: ("irrelevant", "Question regarding whether store accepts damaged tender; unrelated"),
    23: ("irrelevant", "Compliment to customer service staff; unrelated"),
    24: ("irrelevant", "Stock request for vegan brand; unrelated"),

    # Query 1542038: fan asking Tesco for a Twitter follow
    25: ("irrelevant", "Query about replacement charger; unrelated"),
    26: ("irrelevant", "Complaint about noisy shopping centre fan unit; literal polysemy on 'fan'"),
    27: ("irrelevant", "Inquiry about self-scan coin change; unrelated"),
    28: ("partially_relevant", "Inquiry about Tesco following user on Twitter; Tesco explains follow policy for DMs"),
    29: ("partially_relevant", "Customer praise and social banter on Twitter; Tesco friendly reply"),
    30: ("irrelevant", "Customer thanking customer service team; unrelated"),

    # Query 1837286: questioning Tesco Direct partner markups and delivery fees
    31: ("irrelevant", "Inquiry regarding car park fines; unrelated"),
    32: ("irrelevant", "Job application scenario inquiry; unrelated"),
    33: ("irrelevant", "Job application right-to-work documentation; unrelated"),
    34: ("irrelevant", "General request for feedback on internal staff investigation; unrelated"),
    35: ("irrelevant", "General store purchase question; unrelated"),
    36: ("partially_relevant", "Tesco Direct website ordering and payment confusion; Tesco troubleshooting"),

    # Query 1983226: pumpkin price overcharge vs reduced price
    37: ("partially_relevant", "Pricing discrepancy between regular and reduced shelves; Tesco explains promotional vs reduced pricing"),
    38: ("partially_relevant", "Customer overcharged compared to guide price; Tesco explains product variant and refund offer"),
    39: ("partially_relevant", "Multibuy deal pricing discrepancy; Tesco clarifies bundle breakdown"),
    40: ("relevant", "Customer overcharged in store vs promotional shelf price; Tesco explains refund via DM with barcode and receipt"),
    41: ("relevant", "Reduced item overcharged at full price on receipt; Tesco requests receipt photo and issues refund"),
    42: ("partially_relevant", "Promotional deal overcharge dispute; Tesco explains range difference"),

    # Query 1280749: purchased milk past expiry date at Hipperholme store
    43: ("relevant", "Customer purchased out-of-date meat; Tesco alerts store manager and issues refund via Moneycard"),
    44: ("irrelevant", "Store charge inquiry; unrelated to product expiry"),
    45: ("irrelevant", "Voucher expiration time inquiry; unrelated to fresh food expiry"),
    46: ("relevant", "Customer bought out-of-date item at store; Tesco requests details to investigate and resolve"),
    47: ("relevant", "Customer milk spoiled despite date code; Tesco logs supplier investigation and requests batch code"),
    48: ("partially_relevant", "Contaminated lunch purchase; Tesco directs customer to store with receipt for refund"),

    # Query 1295208: customer complaining about ruined evening / poor quality item with photo
    49: ("irrelevant", "Product ingredient and allergen inquiry; unrelated to ruined product complaint"),
    50: ("relevant", "Product quality failure ruined meal; Tesco arranges refund gift card and logs supplier investigation"),
    51: ("irrelevant", "Lost colleague card return inquiry; unrelated"),
    52: ("partially_relevant", "General complaint about store failure; Tesco alerts store manager"),
    53: ("partially_relevant", "Complaint regarding poor product quality; Tesco requests batch code and receipt for refund"),
    54: ("partially_relevant", "Angry complaint with photo; Tesco requests details to log incident"),

    # Query 1136861: recalled baby bath seat, inquiring how to return old one
    55: ("irrelevant", "Inquiry about carrot advertising; unrelated to product recall"),
    56: ("irrelevant", "Parking enforcement inquiry; unrelated to product recall"),
    57: ("irrelevant", "Old £1 coin acceptance inquiry; unrelated to recall"),
    58: ("irrelevant", "General complaint about Bathgate store; unrelated"),
    59: ("partially_relevant", "Customer asking about returning item to store for refund; Tesco explains return details"),
    60: ("irrelevant", "Mouldy melon refund; unrelated to product recall"),

    # Query 1788884: stock shortage of sausages and mozzarella burgers at Cheetham Hill
    61: ("irrelevant", "Trolley coin sign inquiry; unrelated to meat stock"),
    62: ("partially_relevant", "Customer complaining about meat shortage in meal; Tesco logs supplier investigation"),
    63: ("relevant", "Customer checking stock availability at specific local store; Tesco checks inventory system"),
    64: ("relevant", "Customer asking about empty shelves and lack of stock; Tesco investigates store delivery schedule"),
    65: ("relevant", "Customer asking why local store stopped stocking specific sausage product; Tesco checks discontinued status"),
    66: ("irrelevant", "Mouldy tomatoes from click & collect; product quality rather than stock"),

    # Query 1132199: compliment for manned checkout cashier vs self-service
    67: ("partially_relevant", "Customer complaining about unassisted self-service checkouts; Tesco explains staffing feedback"),
    68: ("irrelevant", "Coin change mechanism on self-service; unrelated to staff compliment"),
    69: ("partially_relevant", "Customer preferring manned checkouts over self-service; Tesco acknowledges checkout staffing"),
    70: ("irrelevant", "Customer asking for contact regarding poor service; opposite sentiment"),
    71: ("relevant", "Customer sharing positive customer service experience; Tesco relays praise to store colleagues"),
    72: ("irrelevant", "Delivery charge delay complaint; unrelated"),

    # Query 2008437: dirty baby changing room at Surrey Quays (no soap, broken dryer)
    73: ("relevant", "Customer reporting filthy toilet facility with no soap/paper; Tesco alerts duty manager for immediate cleaning"),
    74: ("irrelevant", "Potato salad out of date; unrelated to store facility cleanliness"),
    75: ("irrelevant", "Gnocchi out of stock at Surrey Quays; unrelated to restroom facility"),
    76: ("irrelevant", "Disturbance in store; safety rather than hygiene maintenance"),
    77: ("relevant", "Customer reporting disgusting toilet bowl with no loo roll; Tesco calls store management immediately"),
    78: ("relevant", "Customer reporting dirty gents toilet; Tesco duty manager dispatches cleaning team"),

    # Query 1493001: grocery website shopping lists removed
    79: ("relevant", "Customer complaining about removal of shopping lists from online account; Tesco confirms temporary removal and return"),
    80: ("relevant", "Customer frustrated by removed grocery shopping lists; Tesco explains feature will return and logs feedback"),
    81: ("relevant", "Customer complaining about deleted shopping lists; Tesco confirms feature return"),
    82: ("relevant", "Customer inquiring why shopping lists removed; Tesco explains trial removal and reinstatement plan"),
    83: ("relevant", "Customer angry about list search removal; Tesco logs IT team feedback"),
    84: ("relevant", "Customer asking to bring back shopping lists; Tesco confirms return in near future"),

    # Query 691333: unable to remove unavailable items from grocery basket
    85: ("irrelevant", "Shopping list removal complaint; different technical issue"),
    86: ("partially_relevant", "Direct website checkout glitch preventing purchase; Tesco offers technical phone support"),
    87: ("irrelevant", "Rice Krispies joke tweet; unrelated"),
    88: ("irrelevant", "Customer asking why item discontinued; unrelated to basket removal"),
    89: ("partially_relevant", "Customer complaining about removed grocery features; Tesco troubleshooting"),
    90: ("partially_relevant", "Unavailable items automatically removed from delivery list; Tesco explains out of stock behavior"),
}

csv_path = Path("data/annotation/retrieval_relevance_dev_sample.csv")
df = pd.read_csv(csv_path, dtype=str)
df["sample_id"] = df["sample_id"].astype(int)
df["score"] = df["score"].astype(float)
df["rank"] = df["rank"].astype(int)
df["human_relevance"] = ""
df["relevance_notes"] = ""

for idx, row in df.iterrows():
    sid = row["sample_id"]
    if sid in annotations:
        label, notes = annotations[sid]
        df.at[idx, "human_relevance"] = label
        df.at[idx, "relevance_notes"] = notes

df.to_csv(csv_path, index=False)
print(f"Updated {len(df)} rows in {csv_path}")

# Validate
validated_df = validate_relevance_file(csv_path)
print("Relevance annotation file passed validation successfully!")

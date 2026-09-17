# Data Directory Convention

Place raw data files in this directory (or at the project root for local testing).

## Accepted Formats
- `tesco_tweets.csv` (the Tesco subset of TWCS dataset)

## Schema Expectations
The dataset is expected to be a CSV file with the following columns:
- `tweet_id` (string/int): Unique identifier for each tweet.
- `author_id` (string): Handle/ID of the tweet author (e.g. `Tesco` or numeric customer ID).
- `inbound` (boolean/string): `TRUE` if from customer to brand, `FALSE` if outbound from brand.
- `created_at` (string): Twitter timestamp formatted as `%a %b %d %H:%M:%S %z %Y`.
- `text` (string): Message content.
- `response_tweet_id` (string): (Optional/Unreliable) Reply tweet IDs.
- `in_response_to_tweet_id` (string): Parent tweet ID if this tweet was a reply.
- `conversation_id` (string): Grouping identifier linking tweets in a conversation thread.

> Note: Raw datasets are excluded by `.gitignore` and MUST NOT be committed to git.

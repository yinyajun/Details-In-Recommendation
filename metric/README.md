### EVALUATE METRIC
1. **weighted, per-user loss**

>Paul Covington, Jay Adams, Emre Sargin Google, Deep Neural Networks for YouTube Recommendations, http://dl.acm.org/citation.cfm?doid=2959100.2959190

>The value shown for each configuration (“weighted, per-user loss”) was obtained by considering both positive (clicked) and negative (unclicked) impressions shown to a user on a single page. We first score these two impressions with our model. If the negative impression receives a higher score than the posi- tive impression, then we consider the positive impression’s watch time to be mispredicted watch time. Weighted, per- user loss is then the total amount mispredicted watch time as a fraction of total watch time over heldout impression pairs.

* using Spark to group by each user, get their labels, predictions and weight(watch time, see paper4.2), respectively
* weighted loss: construct (pos, neg) pair and then calc metric based on pairs for each user.

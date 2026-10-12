"""F0.5 per-entity metric (extracted from notebook cell 17) — reference for self-scoring."""

def f_beta_per_entity(pred_ids, true_ids, beta=0.5):
    """F_beta for ONE S1 entity. pred_ids / true_ids are sets of S2/S3 IDs.
    Singleton rule: empty-vs-empty scores 1.0; predicting anything for a true
    singleton scores 0.0; missing all matches for a true cluster scores 0.0."""
    pred, true = set(pred_ids), set(true_ids)
    if not true and not pred:
        return 1.0                       # correct singleton
    if not true or not pred:
        return 0.0                       # one side empty, other not
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall    = tp / len(true)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)

def macro_f_beta(pred_map, true_map, beta=0.5):
    """Macro-average over ALL S1 entities in true_map (the evaluation set).
    pred_map / true_map: {s1_id: set(matched ids)}. Missing pred -> empty set."""
    scores = [
        f_beta_per_entity(pred_map.get(s1, set()), true_ids, beta)
        for s1, true_ids in true_map.items()
    ]
    return float(np.mean(scores)), len(scores)

# ---- sanity checks against the spec's worked example ----
# GT = {S2-047, S2-193, S3-812}, pred same 3 -> P=1,R=1 -> 1.0
print("exact match      :", round(f_beta_per_entity(
    {"S2-047","S2-193","S3-812"}, {"S2-047","S2-193","S3-812"}), 3))
# INSTRUCTIONS example: pred 3, truth 3, 2 correct -> P=2/3,R=1 -> 0.714
print("P=2/3,R=1        :", round(f_beta_per_entity(
    {"S2-047","S2-193","S3-812"}, {"S2-047","S2-193"}), 3))
# correct singleton -> 1.0 ; false merge on singleton -> 0.0
print("singleton correct:", f_beta_per_entity(set(), set()))
print("singleton merged :", f_beta_per_entity({"S2-1"}, set()))

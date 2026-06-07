"""Demographics stub estimator + aggregation."""
from vision.demographics import StubDemographicsEstimator, aggregate


def test_stub_reads_attrs():
    est = StubDemographicsEstimator()
    out = est.estimate({"gender": "female", "age_bucket": "20-34"})
    assert out["gender"] == "female" and out["age_bucket"] == "20-34"
    assert out["gender_conf"] == 1.0


def test_aggregate_counts():
    ests = [
        {"gender": "female", "age_bucket": "20-34"},
        {"gender": "male", "age_bucket": "20-34"},
        {"gender": "female", "age_bucket": "55+"},
    ]
    agg = aggregate(ests)
    assert agg["n"] == 3
    assert agg["gender"] == {"female": 2, "male": 1}
    assert agg["age_bucket"] == {"20-34": 2, "55+": 1}

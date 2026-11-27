"""An attester for overtime-pay.md (rag-titan reads bundles; it never runs this file)."""


def attest(receipt):
    return {"ok": receipt.get("formula") == "hours * hourly_rate * 1.25"}

"""
Unit tests to verify that OCR text parsing and regex unit standardization
work flawlessly according to Amazon guidelines.
"""

from ocr_extractor import extract_numbers_with_units, UNIT_MAP

def test_extraction():
    test_cases = [
        ("Net Weight: 500g, Pack of 2", "item_weight", "500 gram"),
        ("Shampoo bottle capacity 250ml", "volume", "250 millilitre"),
        ("Extra Virgin Olive Oil 1.5 LTR", "volume", "1.5 litre"),
        ("Heavy duty wire length 10.5 METRE", "width", "10.5 metre"),
        ("Battery voltage 12 V and 60W bulb", "voltage", "12 volt"),
        ("Organic almonds 2.2 lbs pouch", "item_weight", "2.2 pound"),
        ("Coffee beans 500 gm sealed", "item_weight", "500 gram"),
    ]

    print("=" * 60)
    print("Testing Amazon Entity Extraction & Unit Normalizer")
    print("=" * 60)
    
    all_passed = True
    for text, entity, expected in test_cases:
        candidates = extract_numbers_with_units(text, target_entity=entity)
        best = candidates[0]["formatted"] if candidates else "NONE"
        status = "PASS" if best == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"[{status}] Input Text : '{text}'")
        print(f"       Entity     : '{entity}'")
        print(f"       Expected   : '{expected}'")
        print(f"       Extracted  : '{best}'")
        print("-" * 60)

    if all_passed:
        print("[SUCCESS] ALL UNIT EXTRACTION TESTS PASSED PERFECTLY!")
    else:
        print("[ERROR] Some tests failed.")

if __name__ == "__main__":
    test_extraction()

import re


def parse_price(raw_price_string: str) -> float:
    """
    Standardizes Zameen price strings to numerical PKR values.
    Handles 'Crore', 'Lakh', 'Million', 'Thousand', and numeric ranges.
    """
    if not raw_price_string or not isinstance(raw_price_string, str):
        return 0.0
    raw_price = raw_price_string.lower().strip()

    if 'crore' in raw_price:
        components = raw_price.split('crore')

        crore_matches = re.findall(r'[\d\.]+', components[0])
        crore_val = (
            float(crore_matches[0]) if crore_matches else 0.0
        )

        lakh_val = 0.0
        if len(components) > 1 and 'lakh' in components[1]:
            lakh_matches = re.findall(r'[\d\.]+', components[1])
            lakh_val = (
                float(lakh_matches[0]) if lakh_matches else 0.0
            )

        return (crore_val * 10000000.0) + (lakh_val * 100000.0)

    if 'lakh' in raw_price:
        lakh_matches = re.findall(r'[\d\.]+', raw_price)
        parsed_val = (
            float(lakh_matches[0]) if lakh_matches else 0.0
        )
        return parsed_val * 100000.0

    if 'million' in raw_price:
        million_matches = re.findall(r'[\d\.]+', raw_price)
        parsed_val = (
            float(million_matches[0]) if million_matches else 0.0
        )
        return parsed_val * 1000000.0

    if 'thousand' in raw_price:
        thousand_matches = re.findall(r'[\d\.]+', raw_price)
        parsed_val = (
            float(thousand_matches[0]) if thousand_matches else 0.0
        )
        return parsed_val * 1000.0

    regex_numeric_matches = re.findall(r'[\d\.]+', raw_price)
    if regex_numeric_matches:
        return float(regex_numeric_matches[0])
    return 0.0


def parse_area(raw_area_string: str) -> float:
    """
    Standardizes raw Zameen property area strings into square feet.
    Handles 'Kanal', 'Marla', and 'Sq. Yd.' conversions.
    """
    if not raw_area_string or not isinstance(raw_area_string, str):
        return 0.0
    raw_area = raw_area_string.lower().strip()

    if 'kanal' in raw_area:
        kanal_matches = re.findall(r'[\d\.]+', raw_area)
        parsed_val = (
            float(kanal_matches[0]) if kanal_matches else 0.0
        )
        return parsed_val * 4500.0

    if 'marla' in raw_area:
        marla_matches = re.findall(r'[\d\.]+', raw_area)
        parsed_val = (
            float(marla_matches[0]) if marla_matches else 0.0
        )
        return parsed_val * 225.0

    if (
        'sq. yd.' in raw_area
        or 'sq.yd.' in raw_area
        or 'square yards' in raw_area
    ):
        yard_matches = re.findall(r'[\d\.]+', raw_area)
        parsed_val = (
            float(yard_matches[0]) if yard_matches else 0.0
        )
        return parsed_val * 9.0

    regex_numeric_matches = re.findall(r'[\d\.]+', raw_area)
    if regex_numeric_matches:
        return float(regex_numeric_matches[0])
    return 0.0

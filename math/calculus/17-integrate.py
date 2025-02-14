def poly_integral(poly, C=0):
    if not isinstance(poly, list) or not all(isinstance(i, (int, float)) for i in poly):
        return None
    if not isinstance(C, (int, float)):
        return None
    
    # Integral of each term: divide the coefficient by the new exponent (i + 1)
    result = [C]  # Start with the integration constant
    
    for i in range(len(poly)):
        if poly[i] != 0:  # If the coefficient is non-zero
            new_coeff = poly[i] / (i + 1)
            result.append(new_coeff)
    
    # Remove trailing zeros (since they don't affect the polynomial)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    return result

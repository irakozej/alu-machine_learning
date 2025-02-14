#!/usr/bin/env python3

"function"

def poly_integral(poly, C=0):
    """
    Computes the integral of a polynomial.

    Args:
        poly (list): List of coefficients representing a polynomial.
                     The index represents the power of x.
        C (int, float): Integration constant.

    Returns:
        list: New list of coefficients representing the integral.
              Returns None if input is invalid.
    """
    if not isinstance(poly, list) or not all(isinstance(c, (int, float)) for c in poly):
        return None
    if not isinstance(C, (int, float)):
        return None
    if not poly:  # Handle empty list case
        return [C]

    integral = [C]  # Start with the integration constant
    for i, coef in enumerate(poly):
        new_coef = coef / (i + 1)
        integral.append(int(new_coef) if new_coef.is_integer() else new_coef)

    # Remove trailing zeros to keep the list minimal
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral

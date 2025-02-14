#!/usr/bin/env python3

def poly_integral(poly, C=0):
    if not isinstance(poly, list) or not all(isinstance(c, (int, float)) for c in poly):
        return None
    if not isinstance(C, (int, float)):
        return None
    
    integral = [C]  # Start with the integration constant
    for i, coef in enumerate(poly):
        new_coef = coef / (i + 1)
        integral.append(int(new_coef) if new_coef.is_integer() else new_coef)
    
    # Remove trailing zeros to keep the list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    
    return integral


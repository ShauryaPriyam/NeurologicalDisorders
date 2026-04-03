def get_decision(label):
    if label in ["melanoma", "SCC"]:
        return "HIGH", "Visit hospital immediately"
    elif label in ["BCC"]:
        return "MEDIUM", "Consult doctor soon"
    else:
        return "LOW", "Basic care sufficient"
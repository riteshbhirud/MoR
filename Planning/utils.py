def remove_inner_single_quotes(match):
    content = match.group(1)  # Get the string inside ['']
    cleaned_content = content.replace("'", "")  # Remove inner single quotes
    return f"['{cleaned_content}"
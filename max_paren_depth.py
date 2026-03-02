def max_paren_depth(sentence: str) -> int | str:
    depth = 0
    max_depth = 0

    for ch in sentence:
        if ch == '(':
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif ch == ')':
            depth -= 1
            if depth < 0:
                return "Invalid: unmatched closing parenthesis"

    if depth != 0:
        return "Invalid: unmatched opening parenthesis"

    return max_depth


if __name__ == "__main__":
    tests = [
        "Hell, ( dafasd ( sd) ds9( d(( ))d )d d0) ( dsds)",
        "((()))",
        "(()))((",
        "no parens here",
        "(((",
        "))",
    ]

    for s in tests:
        result = max_paren_depth(s)
        print(f"{s!r}  =>  {result}")
